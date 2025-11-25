import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import time
import asyncio
import openai
import re
import json
from opencc import OpenCC
from googleapiclient.discovery import build

# =========================
# 0. 緩存與工具設定 (Caching & Utils)
# =========================

# 緩存時間設定
CACHE_TTL_SEARCH = 3600          # 1 小時：搜尋結果、影片細節
CACHE_TTL_COMMENTS = 900         # 15 分鐘：留言清單

def _get_cached_value(cache_name: str, key, ttl_seconds: int):
    """獲取緩存數據"""
    cache = st.session_state.setdefault(cache_name, {})
    entry = cache.get(key)
    if entry and (time.time() - entry["ts"] <= ttl_seconds):
        return entry["value"]
    return None

def _set_cached_value(cache_name: str, key, value):
    """寫入緩存數據"""
    cache = st.session_state.setdefault(cache_name, {})
    cache[key] = {"value": value, "ts": time.time()}

# 粵語特徵詞庫 (保持原有邏輯作為第一道快速篩選)
CANTONESE_CHAR_TOKENS = {
    "唔": 1.0, "冇": 1.6, "咗": 1.6, "嘅": 1.6, "啲": 1.2, "嗰": 1.2, "佢": 1.0,
    "喺": 1.6, "嚟": 1.6, "咪": 1.2, "啱": 1.2, "掂": 1.2, "靚": 1.2, "曳": 1.2,
    "攰": 1.2, "咁": 1.0, "噉": 1.0, "得": 0.6, "吖": 0.8, "冧": 1.0, "撚": 1.2,
    "仆": 1.2, "屌": 1.2, "嗮": 1.0, "畀": 0.8, "揸": 1.0
}
CANTONESE_PARTICLES = ["啦", "囉", "喎", "咩", "呢", "呀", "嘛", "喇"]
CANTONESE_PHRASES = {
    "好唔好睇": 2.0, "做咩": 1.6, "點解": 1.2, "咩料": 1.6, "算啦": 1.2,
    "得啦": 1.2, "正喎": 1.2, "幾好睇": 1.6, "幾正": 1.2, "好正": 1.0,
    "有啲": 0.8, "嗰啲": 1.2, "呢啲": 1.2, "講真": 0.8, "好似": 0.5
}
ROMANIZATION_RE = re.compile(r"(?i)(?<![A-Za-z])(la|lor|wor|leh|meh|mah|ga|wo|ar)(?=[\s\W]|$)")

def score_cantonese(text: str) -> float:
    """計算粵語特徵分數"""
    if not isinstance(text, str) or not text: return 0.0
    score = 0.0
    for phrase, w in CANTONESE_PHRASES.items():
        if phrase in text: score += text.count(phrase) * w
    for ch, w in CANTONESE_CHAR_TOKENS.items():
        if ch in text: score += text.count(ch) * w
    for p in CANTONESE_PARTICLES:
        if p in text: score += 0.4
    if ROMANIZATION_RE.search(text):
        score += 0.5
    return score

def generate_search_queries(movie_title: str):
    """生成搜尋關鍵字"""
    # 為了節省 API，我們精簡關鍵字，依靠 DeepSeek 後期過濾
    base = [
        f"{movie_title} 影評",
        f"{movie_title} 評價",
        f"{movie_title} 觀後感",
        f"{movie_title} 香港",
        f"{movie_title} review",
        f"{movie_title} reaction"
    ]
    return base

# =========================
# 1. DeepSeek 輔助功能 (AI Helpers)
# =========================

async def check_video_relevance_async(video_list, movie_title, deepseek_client):
    """
    使用 DeepSeek 批量判斷視頻標題是否真的與電影相關。
    video_list: list of dict {'id': vid, 'title': title}
    """
    if not video_list:
        return []
    
    # 構造 Prompt
    titles_text = "\n".join([f"{i}. {v['title']}" for i, v in enumerate(video_list)])
    system_prompt = (
        f"You are a strict movie content filter. Identify which of the following video titles are strictly discussing the movie '{movie_title}'. "
        "Exclude generic vlogs, travel guides, or unrelated news unless they explicitly mention the movie context. "
        "Return a JSON object with a single key 'relevant_indices' containing the list of integer indices (0-based) of the relevant titles."
    )

    try:
        response = await deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": titles_text}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        data = json.loads(response.choices[0].message.content)
        indices = data.get("relevant_indices", [])
        
        valid_ids = []
        for idx in indices:
            if 0 <= idx < len(video_list):
                valid_ids.append(video_list[idx]['id'])
        return valid_ids
    except Exception as e:
        st.warning(f"DeepSeek 視頻過濾失敗，將保留所有視頻: {e}")
        return [v['id'] for v in video_list]

# =========================
# 2. YouTube API 核心 (Cached & Optimized)
# =========================

def search_youtube_videos_optimized(
    keywords, youtube_client, deepseek_client, movie_title,
    max_per_keyword, max_total_videos,
    start_date, end_date
):
    """
    優化版搜尋：
    1. 緩存搜尋結果
    2. 全局總量控制
    3. DeepSeek 標題過濾 (大幅減少無關視頻)
    """
    all_video_ids = set()
    video_meta = {}
    
    search_cache_name = "yt_search_cache"
    
    # 進度條
    progress_text = "正在搜尋 YouTube 影片..."
    my_bar = st.progress(0, text=progress_text)
    
    total_keywords = len(keywords)
    
    for idx, query in enumerate(keywords):
        # 檢查全局上限
        if len(all_video_ids) >= max_total_videos:
            break
            
        cache_key = f"{query}_{start_date}_{end_date}_{max_per_keyword}"
        cached_records = _get_cached_value(search_cache_name, cache_key, CACHE_TTL_SEARCH)
        
        query_records = []
        
        if cached_records is None:
            # 沒緩存，Call API
            try:
                request = youtube_client.search().list(
                    q=query, part="id,snippet", type="video", maxResults=max_per_keyword,
                    publishedAfter=f"{start_date}T00:00:00Z", publishedBefore=f"{end_date}T23:59:59Z",
                    order="relevance", safeSearch="none", relevanceLanguage="zh-Hant", regionCode="HK"
                )
                response = request.execute()
                for item in response.get("items", []):
                    vid = item["id"]["videoId"]
                    snip = item.get("snippet", {})
                    query_records.append({
                        "id": vid,
                        "title": snip.get("title", ""),
                        "channelTitle": snip.get("channelTitle", ""),
                        "publishedAt": snip.get("publishedAt", "")
                    })
                # 寫入緩存
                _set_cached_value(search_cache_name, cache_key, query_records)
            except Exception as e:
                st.warning(f"搜尋 '{query}' 失敗: {e}")
        else:
            query_records = cached_records

        # 收集元數據
        temp_video_list = []
        for rec in query_records:
            if rec["id"] not in all_video_ids:
                video_meta[rec["id"]] = rec
                temp_video_list.append(rec)
        
        # === DeepSeek 介入：過濾無關標題 ===
        if temp_video_list:
            # 異步運行 DeepSeek 過濾
            valid_ids = asyncio.run(check_video_relevance_async(temp_video_list, movie_title, deepseek_client))
            
            # 只添加通過 AI 驗證的 ID
            for vid in valid_ids:
                all_video_ids.add(vid)
                if len(all_video_ids) >= max_total_videos:
                    break
        
        my_bar.progress((idx + 1) / total_keywords, text=f"搜尋中... 已找到 {len(all_video_ids)} 部相關影片")

    my_bar.empty()
    return list(all_video_ids), video_meta

def fetch_channel_details_cached(channel_ids, youtube_client):
    """緩存版：獲取頻道地區資訊"""
    channel_country = {}
    channels_to_fetch = []
    cache_name = "yt_channel_cache"
    
    # 先查緩存
    for cid in channel_ids:
        cached = _get_cached_value(cache_name, cid, CACHE_TTL_SEARCH) # 頻道資訊可緩存久一點
        if cached is not None:
            channel_country[cid] = cached
        else:
            channels_to_fetch.append(cid)
            
    # 批量抓取未緩存的
    if channels_to_fetch:
        for i in range(0, len(channels_to_fetch), 50):
            chunk = channels_to_fetch[i:i+50]
            try:
                resp = youtube_client.channels().list(
                    part="brandingSettings", id=",".join(chunk)
                ).execute()
                for item in resp.get("items", []):
                    cid = item.get("id")
                    country = item.get("brandingSettings", {}).get("channel", {}).get("country", "Unknown")
                    channel_country[cid] = country
                    _set_cached_value(cache_name, cid, country)
            except Exception:
                pass
                
    return channel_country

def get_all_comments_optimized(
    video_ids, youtube_client, 
    max_per_video, max_total_comments,
    video_meta, channel_country_map
):
    """
    優化版留言抓取：
    1. 緩存留言
    2. 全局總量控制
    3. 智能標記視頻來源 (HK Score)
    """
    all_comments = []
    comments_cache_name = "yt_comments_cache"
    
    progress_bar = st.progress(0, text="抓取留言中...")
    total_videos = len(video_ids)
    
    # 獲取視頻詳情以得到 channelId (為了查地區)
    video_channel_map = {}
    # 這裡簡化處理，假設 video_ids 已經經過篩選。
    # 為了省配額，我們只對真正要抓留言的視頻去查 channelId
    # 實際操作中，videos.list 消耗較小 (1 unit)，可以批量做
    
    # 批量獲取 video details (channelId)
    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i:i+50]
        try:
            resp = youtube_client.videos().list(
                part="snippet", id=",".join(chunk)
            ).execute()
            for item in resp.get("items", []):
                video_channel_map[item["id"]] = item["snippet"]["channelId"]
        except: pass

    for i, vid in enumerate(video_ids):
        # 全局上限檢查
        if len(all_comments) >= max_total_comments:
            st.info(f"已達到全局留言上限 ({max_total_comments})，停止抓取。")
            break

        cache_key = f"comments_{vid}_{max_per_video}"
        cached_comments = _get_cached_value(comments_cache_name, cache_key, CACHE_TTL_COMMENTS)
        
        video_comments = []
        
        if cached_comments is not None:
            video_comments = cached_comments
        else:
            # Call API
            try:
                request = youtube_client.commentThreads().list(
                    part="snippet", videoId=vid, textFormat="plainText",
                    order="relevance", maxResults=min(100, max_per_video)
                )
                response = request.execute()
                for item in response.get("items", []):
                    if len(video_comments) >= max_per_video: break
                    comm = item["snippet"]["topLevelComment"]["snippet"]
                    video_comments.append({
                        "comment_text": comm.get("textDisplay", ""),
                        "published_at": comm.get("publishedAt", ""),
                        "like_count": comm.get("likeCount", 0)
                    })
                # 寫入緩存
                _set_cached_value(comments_cache_name, cache_key, video_comments)
            except Exception:
                pass # 評論可能被關閉
        
        # 整合數據
        cid = video_channel_map.get(vid)
        country = channel_country_map.get(cid, "Unknown")
        title = video_meta.get(vid, {}).get("title", "")
        
        # 計算簡單的 HK Score (用於後續智能語言篩選)
        is_hk_source = (country == "HK") or any(k in title for k in ["香港", "粵語", "廣東話", "HK"])
        
        for c in video_comments:
            c_copy = c.copy()
            c_copy.update({
                "video_id": vid,
                "video_title": title,
                "video_url": f"https://www.youtube.com/watch?v={vid}",
                "is_hk_source": is_hk_source
            })
            all_comments.append(c_copy)
            if len(all_comments) >= max_total_comments: break
            
        progress_bar.progress((i + 1) / total_videos, text=f"抓取留言... ({len(all_comments)}/{max_total_comments})")

    progress_bar.empty()
    return pd.DataFrame(all_comments)

# =========================
# 3. DeepSeek 分析與智能語言過濾
# =========================

async def analyze_comment_deepseek_smart(row, deepseek_client, semaphore):
    """
    DeepSeek 核心分析函數：
    同時做：情感分析 + 主題分類 + 智能語言/相關性過濾
    """
    text = row["comment_text"]
    is_hk_source = row["is_hk_source"]
    
    # 智能語言邏輯：
    # 如果視頻來源是香港 (is_hk_source=True)，我們對英文留言寬容 (可能是香港人講英文)。
    # 如果視頻來源不明，我們對英文留言嚴格 (可能是外國人亂入)，需要強制粵語特徵。
    
    system_prompt = (
        "You are a Hong Kong movie analyst. Analyze the comment for the movie. "
        "Output JSON with keys: 'sentiment' (Positive/Negative/Neutral), "
        "'topic' (Plot/Acting/Action/Visuals/Overall/N/A), "
        "'is_relevant_hk_audience' (boolean). "
        "\n\n"
        "Rules for 'is_relevant_hk_audience':\n"
        "1. If the comment is in Cantonese/Traditional Chinese, set True.\n"
        f"2. If the comment is in English: Set {str(is_hk_source).lower()} (based on video source context). "
        "However, if the English comment explicitly mentions Hong Kong cultural context, override to True.\n"
        "3. If Simplified Chinese: Set False unless it uses Cantonese slang.\n"
        "4. If unrelated/spam: Set False."
    )

    async with semaphore:
        try:
            response = await deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            return json.loads(response.choices[0].message.content)
        except:
            return {"sentiment": "Error", "topic": "Error", "is_relevant_hk_audience": False}

async def run_deepseek_analysis(df, deepseek_client):
    semaphore = asyncio.Semaphore(50) # 並發控制
    tasks = []
    
    # 將 DataFrame 轉為 list of dict 方便處理
    rows = df.to_dict('records')
    
    for row in rows:
        tasks.append(analyze_comment_deepseek_smart(row, deepseek_client, semaphore))
    
    progress_bar = st.progress(0, text="AI 智能分析與過濾中...")
    results = []
    for i, f in enumerate(asyncio.as_completed(tasks)):
        res = await f
        results.append(res)
        progress_bar.progress((i + 1) / len(rows))
    
    progress_bar.empty()
    
    # 合併結果，注意異步返回順序可能亂，這裡簡單處理假設順序一致 (as_completed 不保證順序，需修正)
    # 修正：使用 gather 保證順序
    results = await asyncio.gather(*[analyze_comment_deepseek_smart(row, deepseek_client, semaphore) for row in rows])
    
    return pd.DataFrame(results)

# =========================
# 4. 主流程
# =========================

def main_analysis_process(
    movie_title, start_date, end_date, yt_api_key, deepseek_api_key,
    max_per_keyword, max_total_videos, max_per_video, max_total_comments
):
    # 初始化 Clients
    youtube = build("youtube", "v3", developerKey=yt_api_key)
    deepseek = openai.AsyncOpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com/v1")
    
    # 1. 搜尋與過濾 (含 DeepSeek 標題過濾)
    keywords = generate_search_queries(movie_title)
    video_ids, video_meta = search_youtube_videos_optimized(
        keywords, youtube, deepseek, movie_title,
        max_per_keyword, max_total_videos, start_date, end_date
    )
    
    if not video_ids:
        return None, "找不到相關影片 (經 AI 嚴格標題過濾)。"
    
    st.info(f"AI 過濾後保留 {len(video_ids)} 部高度相關影片，開始分析來源...")

    # 2. 獲取頻道地區 (用於智能語言判斷)
    # 為了省配額，我們需要先拿到 channel IDs
    # 這裡稍微取巧，先不 call videos.list 拿 channelId，等到抓 comment 時順便拿，或者只對前 N 個拿
    # 為了準確性，還是得拿。使用緩存優化。
    temp_vids_chunk = video_ids[:max_total_videos] # 再次確保不超量
    
    # 快速獲取 Channel IDs (消耗 1 unit per 50 videos)
    vid_to_cid = {}
    for i in range(0, len(temp_vids_chunk), 50):
        try:
            resp = youtube.videos().list(part="snippet", id=",".join(temp_vids_chunk[i:i+50])).execute()
            for item in resp.get("items", []):
                vid_to_cid[item["id"]] = item["snippet"]["channelId"]
        except: pass
        
    channel_ids = list(set(vid_to_cid.values()))
    channel_country_map = fetch_channel_details_cached(channel_ids, youtube)
    
    # 3. 抓取留言 (含全局總量控制)
    df_comments = get_all_comments_optimized(
        temp_vids_chunk, youtube, max_per_video, max_total_comments,
        video_meta, channel_country_map
    )
    
    if df_comments.empty:
        return None, "找不到任何留言。"
        
    st.info(f"已抓取 {len(df_comments)} 則原始留言，正在進行 DeepSeek 智能語義分析與篩選...")
    
    # 4. DeepSeek 終極分析 (情感 + 智能語言過濾)
    analysis_df = asyncio.run(run_deepseek_analysis(df_comments, deepseek))
    
    # 合併並過濾
    final_df = pd.concat([df_comments, analysis_df], axis=1)
    
    # 應用 "is_relevant_hk_audience" 過濾
    original_count = len(final_df)
    final_df = final_df[final_df["is_relevant_hk_audience"] == True].copy()
    filtered_count = len(final_df)
    
    st.success(f"分析完成！AI 剔除了 {original_count - filtered_count} 則非目標受眾(純外語/簡體/無關)留言，保留 {filtered_count} 則有效粵語/香港觀點留言。")
    
    # 格式化時間供圖表使用
    final_df["published_at"] = pd.to_datetime(final_df["published_at"])
    
    return final_df, None

# =========================
# 5. Streamlit UI
# =========================

st.set_page_config(page_title="YouTube 電影評論 AI 分析 (Pro)", layout="wide")
st.title("🎬 YouTube 電影評論 AI 分析 (Pro 版)")
st.markdown("### 🚀 智能省流版：DeepSeek 深度介入 + 全局配額控制")

with st.sidebar:
    st.header("設定")
    yt_api_key = st.text_input("YouTube API Key", type='password')
    deepseek_api_key = st.text_input("DeepSeek API Key", type='password')
    
    st.divider()
    st.subheader("配額與過濾控制")
    max_total_videos = st.number_input("全局最大影片分析數", 10, 200, 50, help="達到此數量後停止搜尋，節省配額")
    max_total_comments = st.number_input("全局最大留言分析數", 50, 2000, 500, help="達到此數量後停止抓取，節省配額")
    
    st.divider()
    max_per_keyword = st.slider("單關鍵字搜尋上限", 10, 50, 20)
    max_per_video = st.slider("單影片留言上限", 20, 100, 50)

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    movie_title = st.text_input("電影名稱", value="九龍城寨之圍城")
with col2:
    start_date = st.date_input("開始日期", value=datetime.today() - timedelta(days=30))
with col3:
    end_date = st.date_input("結束日期", value=datetime.today())

if st.button("🚀 開始智能分析", type="primary"):
    if not all([movie_title, yt_api_key, deepseek_api_key]):
        st.error("請填寫所有 API Key 和電影名稱")
    else:
        with st.spinner("正在調用 AI 進行多層次分析... (搜尋結果將被緩存)"):
            df_result, err = main_analysis_process(
                movie_title, str(start_date), str(end_date),
                yt_api_key, deepseek_api_key,
                max_per_keyword, max_total_videos, max_per_video, max_total_comments
            )
            
        if err:
            st.error(err)
        else:
            # =========================
            # Visualization (保持原樣)
            # =========================
            st.divider()
            st.subheader("📊 分析結果可視化")
            
            # 1. 情感分佈
            sentiments_order = ['Positive', 'Negative', 'Neutral']
            colors_map = {'Positive': '#5cb85c', 'Negative': '#d9534f', 'Neutral': '#f0ad4e'}
            
            c1, c2 = st.columns(2)
            with c1:
                vc = df_result['sentiment'].value_counts()
                fig1 = px.pie(values=vc.values, names=vc.index, title='整體情感分佈', 
                              color=vc.index, color_discrete_map=colors_map)
                st.plotly_chart(fig1, use_container_width=True)
            
            with c2:
                # 主題分佈
                if 'topic' in df_result.columns:
                    topic_counts = df_result['topic'].value_counts()
                    fig2 = px.bar(x=topic_counts.index, y=topic_counts.values, title='評論主題分佈',
                                  labels={'x': '主題', 'y': '數量'})
                    st.plotly_chart(fig2, use_container_width=True)

            # 2. 時間趨勢
            if not df_result.empty:
                df_result['date'] = df_result['published_at'].dt.date
                daily = df_result.groupby(['date', 'sentiment']).size().unstack().fillna(0)
                daily = daily.reindex(columns=sentiments_order).dropna(axis=1, how='all')
                
                if not daily.empty:
                    daily_long = daily.reset_index().melt(id_vars='date', var_name='sentiment', value_name='count')
                    fig3 = px.line(daily_long, x='date', y='count', color='sentiment',
                                   title='每日情感趨勢', color_discrete_map=colors_map)
                    st.plotly_chart(fig3, use_container_width=True)

            # 3. 數據表
            st.subheader("📝 詳細數據 (含來源標記)")
            st.dataframe(
                df_result[['sentiment', 'topic', 'comment_text', 'video_title', 'is_hk_source', 'published_at']], 
                use_container_width=True
            )
            
            csv = df_result.to_csv(index=False).encode('utf-8-sig')
            st.download_button("📥 下載 CSV", csv, "analysis_result.csv", "text/csv")
