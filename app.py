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

# ... (前面的緩存函數 _get_cached_value, _set_cached_value 保持不變) ...

# =========================
# 0. 工具設定 (更新)
# =========================

CACHE_TTL_SEARCH = 3600
CACHE_TTL_COMMENTS = 900

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
    return [
        f"{movie_title}",
        f"{movie_title} 影評",
        f"{movie_title} 觀後感",
        f"{movie_title} review",
        f"{movie_title} 電影"
    ]

# =========================
# 1. YouTube API 核心 (含政治/無關內容過濾)
# =========================

def search_youtube_videos_smart(
    keywords, youtube_client, movie_title,
    max_per_keyword, max_total_videos,
    start_date, end_date,
    negative_keywords_list  # 新增：負面關鍵詞列表
):
    all_video_ids = set()
    video_meta = {}
    search_cache_name = "yt_search_smart_filtered_cache"
    
    progress_text = "正在搜尋並過濾無關/政治影片..."
    my_bar = st.progress(0, text=progress_text)
    
    # 1. 準備正向關鍵詞 (電影名拆分)
    # 例如 "九龍城寨之圍城" -> ["九龍城寨", "圍城"]
    title_keywords = [k for k in re.split(r'\s+|：|:|,|，', movie_title) if len(k) > 1]
    
    # 2. 準備負面關鍵詞 (硬編碼基礎 + 用戶輸入)
    # 這些詞出現在標題中通常代表是時政新聞而非影評
    base_negative_keywords = [
        "新聞", "直播", "施政報告", "習近平", "中共", "共產黨", 
        "特首", "李家超", "立法會", "示威", "政治", "政經", 
        "大紀元", "文昭", "江峰", "天亮時分", "時事", "財經"
    ]
    # 合併用戶定義的排除詞
    final_negative_keywords = list(set(base_negative_keywords + negative_keywords_list))

    for idx, query in enumerate(keywords):
        if len(all_video_ids) >= max_total_videos: break
            
        cache_key = f"{query}_{start_date}_{end_date}_{max_per_keyword}_filtered"
        cached_records = _get_cached_value(search_cache_name, cache_key, CACHE_TTL_SEARCH)
        
        query_records = []
        
        if cached_records is None:
            try:
                request = youtube_client.search().list(
                    q=query, part="id,snippet", type="video", maxResults=max_per_keyword,
                    publishedAfter=f"{start_date}T00:00:00Z", publishedBefore=f"{end_date}T23:59:59Z",
                    order="relevance", safeSearch="none", relevanceLanguage="zh-Hant"
                )
                response = request.execute()
                for item in response.get("items", []):
                    vid = item["id"]["videoId"]
                    snip = item.get("snippet", {})
                    title = snip.get("title", "")
                    desc = snip.get("description", "")
                    channel_title = snip.get("channelTitle", "")
                    
                    # === 核心修改：雙重過濾邏輯 ===
                    
                    # A. 負面過濾 (Negative Filter) - 優先級最高
                    # 如果標題或頻道名包含政治敏感詞，直接跳過
                    if any(nk in title for nk in final_negative_keywords) or \
                       any(nk in channel_title for nk in final_negative_keywords):
                        continue 

                    # B. 正向相關性檢查 (Positive Relevance)
                    is_relevant = False
                    
                    # B1. 標題必須包含至少一個電影核心詞
                    # 這是為了防止 YouTube 推送完全無關的 "猜你喜歡"
                    if any(tk.lower() in title.lower() for tk in title_keywords):
                        is_relevant = True
                    
                    # B2. 如果標題沒有核心詞，但描述裡有完整電影名，也可以接受 (防止標題黨)
                    elif movie_title.lower() in desc.lower():
                        is_relevant = True
                        
                    if is_relevant:
                        query_records.append({
                            "id": vid,
                            "title": title,
                            "channelTitle": channel_title,
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
        
        my_bar.progress((idx + 1) / len(keywords), text=f"搜尋中... 已過濾政治/無關內容，保留: {len(all_video_ids)} 部")

    my_bar.empty()
    return list(all_video_ids), video_meta

# ... (get_all_comments_cached 函數保持不變，使用上一版的動態調整邏輯) ...
def get_all_comments_cached(video_ids, youtube_client, max_per_video, max_total_comments, video_meta):
    all_comments = []
    comments_cache_name = "yt_comments_cache_v4" # Update version
    
    progress_bar = st.progress(0, text="抓取留言中...")
    
    # 動態調整：影片少則抓更多評論
    if len(video_ids) > 0 and len(video_ids) < 5:
        adjusted_max_per_video = max_per_video * 4 # 提升倍率
        st.caption(f"⚠️ 經過濾後影片來源較少，自動將單一影片留言抓取上限大幅提升至 {adjusted_max_per_video} 則")
    else:
        adjusted_max_per_video = max_per_video

    for i, vid in enumerate(video_ids):
        if len(all_comments) >= max_total_comments: break

        cache_key = f"comments_{vid}_{adjusted_max_per_video}"
        cached_comments = _get_cached_value(comments_cache_name, cache_key, CACHE_TTL_COMMENTS)
        
        video_comments = []
        if cached_comments is not None:
            video_comments = cached_comments
        else:
            try:
                request = youtube_client.commentThreads().list(
                    part="snippet", videoId=vid, textFormat="plainText",
                    order="relevance", maxResults=min(100, adjusted_max_per_video)
                )
                response = request.execute()
                for item in response.get("items", []):
                    if len(video_comments) >= adjusted_max_per_video: break
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
# 2. DeepSeek 分析 (放宽标准 & 调试版)
# =========================

async def analyze_comment_deepseek_v2(row, client, semaphore):
    """
    單個評論分析函數
    """
    text = row["comment_text"]
    video_title = row.get("video_title", "")
    
    # === 修改 1: 放寬 System Prompt，並要求返回 reason ===
    system_prompt = (
        "You are a lenient movie analyst for the Hong Kong market. "
        f"The comment is from a video: '{video_title}'. "
        "Analyze the comment. "
        "Output JSON with keys: "
        "'sentiment' (Positive/Negative/Neutral), "
        "'keywords' (Extract 1-2 main keywords), "
        "'is_target_audience' (boolean), "
        "'reason' (Short string explaining why is_target_audience is true/false). "
        "\n\n"
        "Rules for 'is_target_audience' (BE LENIENT):\n"
        "1. **TRUE** if it expresses ANY opinion about the movie, actors, or the video content.\n"
        "2. **TRUE** even if it is Simplified Chinese, as long as it is readable.\n"
        "3. **TRUE** even if it mentions politics, AS LONG AS it relates to the movie's themes or context.\n"
        "4. **TRUE** for short comments like 'Good', 'Bad', 'Support'.\n"
        "5. **FALSE** ONLY if it is clearly spam, advertisements, or completely gibberish/unrelated nonsense."
    )

    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                timeout=20 
            )
            content = response.choices[0].message.content
            if not content:
                # 如果內容為空，默認保留 (True)，以免誤刪
                return {"sentiment": "Neutral", "keywords": "Empty Response", "is_target_audience": True, "reason": "API returned empty"}
            
            result = json.loads(content)
            
            # === 修改 2: 強制類型轉換，防止 AI 返回字符串 "true" 導致過濾失敗 ===
            is_target = result.get("is_target_audience", True)
            if isinstance(is_target, str):
                is_target = is_target.lower() == 'true'
            result["is_target_audience"] = is_target
            
            # 確保有 reason 字段
            if "reason" not in result:
                result["reason"] = "No reason provided"
                
            return result

        except json.JSONDecodeError:
            # JSON 解析失敗，通常是因為 AI 輸出了多餘文字，我們選擇保留這條評論人工查看
            return {"sentiment": "Neutral", "keywords": "JSON Error", "is_target_audience": True, "reason": "JSON Parse Error"}
        except Exception as e:
            # 其他錯誤 (網絡等)，也默認保留 (True)
            return {"sentiment": "Error", "keywords": "System Error", "is_target_audience": True, "reason": f"Error: {str(e)}"}

async def run_deepseek_analysis(df, api_key):
    """
    主異步控制器
    """
    client = openai.AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
    
    # 降低併發數保證穩定性
    semaphore = asyncio.Semaphore(10) 
    rows = df.to_dict('records')
    
    progress_bar = st.progress(0, text="AI 正在深度分析 (已放寬過濾標準)...")
    
    tasks = [analyze_comment_deepseek_v2(row, client, semaphore) for row in rows]
    
    results = []
    total = len(tasks)
    
    for i, task in enumerate(asyncio.as_completed(tasks)):
        await task 
        progress_bar.progress((i + 1) / total)
    
    results_raw = await asyncio.gather(*tasks, return_exceptions=True)
    
    await client.close()
    progress_bar.empty()
    
    clean_results = []
    for res in results_raw:
        if isinstance(res, Exception):
            # 嚴重錯誤時，保留數據並標記
            clean_results.append({"sentiment": "Error", "keywords": "Critical Fail", "is_target_audience": True, "reason": str(res)})
        else:
            clean_results.append(res)
            
    return pd.DataFrame(clean_results)
# ... (前面代码不变) ...
    
    # 3. AI 分析
    try:
        analysis_df = asyncio.run(run_deepseek_analysis(df_comments, deepseek_api_key))
    except Exception as e:
        return None, f"AI 分析過程中發生錯誤: {str(e)}"

    final_df = pd.concat([df_comments, analysis_df], axis=1)
    
    # 4. 篩選
    original_len = len(final_df)
    
    # 確保是布林值
    final_df["is_target_audience"] = final_df["is_target_audience"].fillna(True).astype(bool)
    
    # === 调试模式：先不删除，看看数据 ===
    # 如果你发现还是 0，可以暂时注释掉下面这一行过滤代码，看看 reason 写的是什么
    final_df_filtered = final_df[final_df["is_target_audience"] == True].copy()
    
    filtered_len = len(final_df_filtered)
    
    if filtered_len == 0 and original_len > 0:
        st.warning("警告：所有評論都被 AI 過濾掉了。以下是前 5 條被拒絕的原因，請檢查：")
        st.write(final_df[["comment_text", "reason"]].head())
        # 為了讓你能看到數據，這種情況下我們返回原始數據
        return final_df, "所有數據被過濾，已返回原始數據供調試。"

    st.success(f"分析完成！原始 {original_len} 則，有效 {filtered_len} 則。")
    
    final_df_filtered["published_at"] = pd.to_datetime(final_df_filtered["published_at"])
    
    # 返回包含 reason 的數據，方便你在 UI 上查看
    return final_df_filtered, None
# =========================
# 4. UI
# =========================

st.set_page_config(page_title="YouTube 影評分析 (Anti-Spam)", layout="wide")
st.title("🎬 YouTube 影評分析 (智能過濾版)")
st.markdown("### 特點：智能搜尋 | 🚫 自動過濾政治/新聞影片 | 繁體/粵語識別")

with st.sidebar:
    st.header("設定")
    yt_api_key = st.text_input("YouTube API Key", type='password')
    deepseek_api_key = st.text_input("DeepSeek API Key", type='password')
    st.divider()
    max_total_videos = st.number_input("最大影片搜尋數", 10, 100, 50)
    max_total_comments = st.number_input("最大分析留言數", 50, 2000, 500)
    
    st.divider()
    st.subheader("🚫 排除關鍵詞 (防止政治干擾)")
    default_neg = "新聞, 直播, 習近平, 中共, 政治"
    user_neg_input = st.text_area("輸入要排除的詞 (逗號分隔)", value=default_neg, help="若標題包含這些詞，將直接忽略該影片")
    negative_keywords = [x.strip() for x in user_neg_input.split(",") if x.strip()]

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    movie_title = st.text_input("電影名稱", value="九龍城寨") 
with col2:
    start_date = st.date_input("開始", value=datetime.today() - timedelta(days=90))
with col3:
    end_date = st.date_input("結束", value=datetime.today())

if st.button("🚀 開始分析", type="primary"):
    if not all([movie_title, yt_api_key, deepseek_api_key]):
        st.error("請填寫所有欄位")
    else:
        with st.spinner("正在搜尋並執行雙重過濾 (關鍵詞 + AI)..."):
            df_result, err = main_process(
                movie_title, str(start_date), str(end_date),
                yt_api_key, deepseek_api_key,
                20, max_total_videos, 50, max_total_comments,
                negative_keywords
            )
            
        if err:
            st.error(err)
        else:
            st.divider()
            
            # 簡單展示結果 (保留原有的可視化代碼結構)
            st.subheader("🔥 熱門評論關鍵詞")
            # ... (此處可視化代碼與上一版相同，省略以節省篇幅) ...
            all_keywords = []
            for item in df_result['keywords']:
                if isinstance(item, str):
                    words = [w.strip() for w in re.split(r'[，,、\s]+', item) if len(w.strip()) > 1]
                    all_keywords.extend(words)
                elif isinstance(item, list):
                    all_keywords.extend([str(w).strip() for w in item if len(str(w).strip()) > 1])
            
            if all_keywords:
                kw_counts = Counter(all_keywords).most_common(15)
                kw_df = pd.DataFrame(kw_counts, columns=['Keyword', 'Count']).sort_values(by='Count')
                fig_kw = px.bar(kw_df, x='Count', y='Keyword', orientation='h', title='Top Keywords')
                st.plotly_chart(fig_kw, use_container_width=True)

            with st.expander("查看詳細數據"):
                st.dataframe(df_result[['sentiment', 'keywords', 'comment_text', 'video_title']])
