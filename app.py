import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import asyncio
import openai
import re
import json
from googleapiclient.discovery import build
from collections import Counter

# =========================
# 0. 緩存與工具設定
# =========================

CACHE_TTL_SEARCH = 3600          # 1 小時
CACHE_TTL_COMMENTS = 900         # 15 分鐘

def _get_cached_value(cache_name: str, key, ttl_seconds: int):
    cache = st.session_state.setdefault(cache_name, {})
    entry = cache.get(key)
    if entry and (time.time() - entry["ts"] <= ttl_seconds):
        return entry["value"]
    return None

def _set_cached_value(cache_name: str, key, value):
    cache = st.session_state.setdefault(cache_name, {})
    cache[key] = {"value": value, "ts": time.time()}

def generate_search_queries(movie_title: str):
    # 為了確保能搜到，我們使用精確匹配的邏輯，但在 API 查詢時還是要給一點廣度
    # 嚴格過濾會在代碼層面做
    return [
        f"{movie_title}",
        f"{movie_title} 影評",
        f"{movie_title} 評價",
        f"{movie_title} 香港",
        f"{movie_title} 粵語"
    ]

# =========================
# 1. YouTube API 核心 (含嚴格標題過濾)
# =========================

def search_youtube_videos_strict(
    keywords, youtube_client, movie_title,
    max_per_keyword, max_total_videos,
    start_date, end_date
):
    all_video_ids = set()
    video_meta = {}
    search_cache_name = "yt_search_strict_cache"
    
    progress_text = "正在搜尋並嚴格過濾影片..."
    my_bar = st.progress(0, text=progress_text)
    
    # 預處理電影標題，轉小寫以進行不區分大小寫的匹配
    target_title_lower = movie_title.strip().lower()
    
    for idx, query in enumerate(keywords):
        if len(all_video_ids) >= max_total_videos: break
            
        cache_key = f"{query}_{start_date}_{end_date}_{max_per_keyword}_strict"
        cached_records = _get_cached_value(search_cache_name, cache_key, CACHE_TTL_SEARCH)
        
        query_records = []
        
        if cached_records is None:
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
                    title = snip.get("title", "")
                    
                    # === 核心修改：100% 嚴格標題匹配 ===
                    # 只有當電影名稱完整出現在標題中才保留
                    if target_title_lower in title.lower():
                        query_records.append({
                            "id": vid,
                            "title": title,
                            "channelTitle": snip.get("channelTitle", ""),
                            "publishedAt": snip.get("publishedAt", "")
                        })
                
                _set_cached_value(search_cache_name, cache_key, query_records)
            except Exception as e:
                st.warning(f"搜尋 '{query}' 失敗: {e}")
        else:
            query_records = cached_records

        for rec in query_records:
            if rec["id"] not in all_video_ids:
                all_video_ids.add(rec["id"])
                video_meta[rec["id"]] = rec
                if len(all_video_ids) >= max_total_videos: break
        
        my_bar.progress((idx + 1) / len(keywords), text=f"搜尋中... 符合嚴格標題條件的影片: {len(all_video_ids)} 部")

    my_bar.empty()
    return list(all_video_ids), video_meta

def get_all_comments_cached(video_ids, youtube_client, max_per_video, max_total_comments, video_meta):
    all_comments = []
    comments_cache_name = "yt_comments_cache_v2"
    
    progress_bar = st.progress(0, text="抓取留言中...")
    
    for i, vid in enumerate(video_ids):
        if len(all_comments) >= max_total_comments: break

        cache_key = f"comments_{vid}_{max_per_video}"
        cached_comments = _get_cached_value(comments_cache_name, cache_key, CACHE_TTL_COMMENTS)
        
        video_comments = []
        if cached_comments is not None:
            video_comments = cached_comments
        else:
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
                _set_cached_value(comments_cache_name, cache_key, video_comments)
            except: pass
        
        title = video_meta.get(vid, {}).get("title", "")
        for c in video_comments:
            c_copy = c.copy()
            c_copy.update({"video_id": vid, "video_title": title})
            all_comments.append(c_copy)
            if len(all_comments) >= max_total_comments: break
            
        progress_bar.progress((i + 1) / len(video_ids), text=f"抓取留言... ({len(all_comments)}/{max_total_comments})")

    progress_bar.empty()
    return pd.DataFrame(all_comments)

# =========================
# 2. DeepSeek 分析 (語言篩選 + 關鍵詞提取)
# =========================

async def analyze_comment_deepseek_v2(row, deepseek_client, semaphore):
    text = row["comment_text"]
    
    # Prompt 策略：
    # 1. 嚴格拒絕英文 (Reject English)
    # 2. 接受粵語、繁體中文 (Accept Cantonese/Traditional)
    # 3. 提取關鍵詞 (Extract Keywords)
    
    system_prompt = (
        "You are a Hong Kong movie analyst. Analyze the comment. "
        "Output JSON with keys: "
        "'sentiment' (Positive/Negative/Neutral), "
        "'keywords' (Extract 1-2 main keywords/short phrases in Traditional Chinese, e.g. '劇情', '古天樂', '打鬥'), "
        "'is_cantonese_target' (boolean). "
        "\n\n"
        "Rules for 'is_cantonese_target':\n"
        "1. **Strictly Set False** if the comment is in English (even if positive).\n"
        "2. Set True if the comment is in Cantonese (contains slang like 唔, 係, 嘅, 佢) or Traditional Chinese.\n"
        "3. If the comment is ambiguous (short Chinese phrases), Set True (give benefit of doubt).\n"
        "4. Set False for spam or unrelated content."
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
            return {"sentiment": "Error", "keywords": "", "is_cantonese_target": False}

async def run_deepseek_analysis(df, deepseek_client):
    semaphore = asyncio.Semaphore(50)
    rows = df.to_dict('records')
    
    # 使用 gather 確保順序一致
    tasks = [analyze_comment_deepseek_v2(row, deepseek_client, semaphore) for row in rows]
    
    progress_bar = st.progress(0, text="AI 正在進行情感分析與關鍵詞提取...")
    
    # 為了顯示進度，我們稍微包裝一下
    results = []
    total = len(tasks)
    for i, task in enumerate(asyncio.as_completed(tasks)):
        await task # 這裡只是為了觸發進度條，實際結果順序由 gather 決定
        progress_bar.progress((i + 1) / total)
    
    # 重新按順序獲取結果
    results = await asyncio.gather(*[analyze_comment_deepseek_v2(row, deepseek_client, semaphore) for row in rows])
    progress_bar.empty()
    
    return pd.DataFrame(results)

# =========================
# 3. 主流程
# =========================

def main_process(movie_title, start_date, end_date, yt_api_key, deepseek_api_key, 
                 max_per_keyword, max_total_videos, max_per_video, max_total_comments):
    
    youtube = build("youtube", "v3", developerKey=yt_api_key)
    deepseek = openai.AsyncOpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com/v1")
    
    # 1. 搜尋 (嚴格標題)
    keywords = generate_search_queries(movie_title)
    video_ids, video_meta = search_youtube_videos_strict(
        keywords, youtube, movie_title,
        max_per_keyword, max_total_videos, start_date, end_date
    )
    
    if not video_ids:
        return None, f"找不到標題包含「{movie_title}」的影片。"
    
    st.info(f"已鎖定 {len(video_ids)} 部標題完全匹配的影片，開始抓取留言...")
    
    # 2. 抓取留言
    df_comments = get_all_comments_cached(video_ids, youtube, max_per_video, max_total_comments, video_meta)
    
    if df_comments.empty:
        return None, "這些影片下沒有找到留言。"
    
    # 3. AI 分析
    analysis_df = asyncio.run(run_deepseek_analysis(df_comments, deepseek))
    final_df = pd.concat([df_comments, analysis_df], axis=1)
    
    # 4. 過濾非粵語/英文
    original_len = len(final_df)
    final_df = final_df[final_df["is_cantonese_target"] == True].copy()
    filtered_len = len(final_df)
    
    st.success(f"分析完成！共抓取 {original_len} 則留言，AI 剔除非粵語/純英文留言後，剩餘 {filtered_len} 則有效數據。")
    
    final_df["published_at"] = pd.to_datetime(final_df["published_at"])
    return final_df, None

# =========================
# 4. Streamlit UI & Visualization
# =========================

st.set_page_config(page_title="YouTube 粵語影評分析", layout="wide")
st.title("🎬 YouTube 粵語影評精準分析")
st.markdown("### 特點：100% 標題匹配 | 剔除英文 | 粵語優先 | 深度可視化")

with st.sidebar:
    st.header("設定")
    yt_api_key = st.text_input("YouTube API Key", type='password')
    deepseek_api_key = st.text_input("DeepSeek API Key", type='password')
    st.divider()
    max_total_videos = st.number_input("最大影片數", 10, 100, 30)
    max_total_comments = st.number_input("最大分析留言數", 50, 1000, 300)

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    movie_title = st.text_input("電影全名 (必須完全匹配)", value="非常盜3") # 測試用例
with col2:
    start_date = st.date_input("開始", value=datetime.today() - timedelta(days=60))
with col3:
    end_date = st.date_input("結束", value=datetime.today())

if st.button("🚀 開始分析", type="primary"):
    if not all([movie_title, yt_api_key, deepseek_api_key]):
        st.error("請填寫所有欄位")
    else:
        with st.spinner("AI 正在全力運算中..."):
            df_result, err = main_process(
                movie_title, str(start_date), str(end_date),
                yt_api_key, deepseek_api_key,
                20, max_total_videos, 50, max_total_comments
            )
            
        if err:
            st.error(err)
        else:
            # ==========================================
            # Visualization 優化部分
            # ==========================================
            st.divider()
            
            # 1. 關鍵詞分析 (Horizontal Bar Chart)
            st.subheader("🔥 熱門評論關鍵詞 (Top Keywords)")
            
            # 處理關鍵詞：DeepSeek 可能返回 list 或 string，需標準化
            all_keywords = []
            for item in df_result['keywords']:
                if isinstance(item, str):
                    # 假設逗號分隔
                    words = [w.strip() for w in re.split(r'[，,、\s]+', item) if len(w.strip()) > 1]
                    all_keywords.extend(words)
                elif isinstance(item, list):
                    all_keywords.extend([str(w).strip() for w in item if len(str(w).strip()) > 1])
            
            if all_keywords:
                kw_counts = Counter(all_keywords).most_common(15)
                kw_df = pd.DataFrame(kw_counts, columns=['Keyword', 'Count'])
                kw_df = kw_df.sort_values(by='Count', ascending=True) # 為了讓 Bar Chart 最高在最上面
                
                fig_kw = px.bar(
                    kw_df, x='Count', y='Keyword', orientation='h',
                    title='Top 15 Most Mentioned Keywords',
                    text='Count',
                    color='Count',
                    color_continuous_scale='Blues'
                )
                fig_kw.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_kw, use_container_width=True)
            else:
                st.info("無法提取足夠的關鍵詞數據。")

            # 2. 情感走勢分析 (Line + Stacked Bar)
            st.subheader("📈 情感趨勢分析 (Sentiment Trend)")
            
            # 數據預處理
            df_result['date'] = df_result['published_at'].dt.date
            sentiments = ['Positive', 'Negative', 'Neutral']
            colors = {'Positive': '#28a745', 'Negative': '#dc3545', 'Neutral': '#ffc107'}
            
            # 聚合數據
            daily_sentiment = df_result.groupby(['date', 'sentiment']).size().reset_index(name='count')
            
            # 確保日期連續性 (可選，為了圖表好看)
            if not daily_sentiment.empty:
                min_date = daily_sentiment['date'].min()
                max_date = daily_sentiment['date'].max()
                all_dates = pd.date_range(min_date, max_date).date
                
                # 建立完整索引
                full_idx = pd.MultiIndex.from_product([all_dates, sentiments], names=['date', 'sentiment'])
                daily_sentiment = daily_sentiment.set_index(['date', 'sentiment']).reindex(full_idx, fill_value=0).reset_index()

                # A. 折線圖 (Line Chart) - 顯示走勢
                fig_line = px.line(
                    daily_sentiment, x='date', y='count', color='sentiment',
                    title='每日情感變化趨勢 (Line Chart)',
                    color_discrete_map=colors,
                    markers=True
                )
                st.plotly_chart(fig_line, use_container_width=True)
                
                # B. 堆疊柱狀圖 (Stacked Bar Chart) - 顯示總量與構成
                fig_stack = px.bar(
                    daily_sentiment, x='date', y='count', color='sentiment',
                    title='每日評論總量與情感構成 (Stacked Bar)',
                    color_discrete_map=colors,
                    barmode='stack'
                )
                st.plotly_chart(fig_stack, use_container_width=True)
            else:
                st.warning("數據不足以生成趨勢圖。")

            # 3. 數據明細
            with st.expander("查看詳細數據 (CSV 下載)"):
                st.dataframe(df_result[['sentiment', 'keywords', 'comment_text', 'video_title', 'published_at']])
                csv = df_result.to_csv(index=False).encode('utf-8-sig')
                st.download_button("📥 下載完整 CSV", csv, "cantonese_analysis.csv", "text/csv")
