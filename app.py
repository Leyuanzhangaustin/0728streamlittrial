# app.py (Accelerated Version)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import asyncio # <<< MODIFIED: 引入 asyncio
import openai # <<< MODIFIED: 確保 openai 已引入

# ========== 1. YouTube Search (No changes) ==========
def search_youtube_videos(keywords, youtube_client, max_per_keyword, start_date, end_date):
    all_video_ids = set()
    for query in keywords:
        try:
            search_response = youtube_client.search().list(
                q=query,
                part='id,snippet',
                type='video',
                maxResults=max_per_keyword,
                publishedAfter=f"{start_date}T00:00:00Z",
                publishedBefore=f"{end_date}T23:59:59Z",
                relevanceLanguage='zh-Hant',
                regionCode='HK'
            ).execute()
            video_ids = [item['id']['videoId'] for item in search_response.get('items', [])]
            all_video_ids.update(video_ids)
            time.sleep(0.5) # Keep a small delay to be nice to YouTube API
        except Exception as e:
            st.warning(f"搜尋關鍵字 '{query}' 時發生錯誤: {e}")
            continue
    return list(all_video_ids)

# ========== 2. Batch Fetch Comments (No changes) ==========
def get_all_comments(video_ids, youtube_client, max_per_video):
    all_comments = []
    for video_id in video_ids:
        try:
            request = youtube_client.commentThreads().list(
                part='snippet', videoId=video_id, textFormat='plainText', maxResults=100
            )
            comments_fetched = 0
            while request and comments_fetched < max_per_video:
                response = request.execute()
                for item in response['items']:
                    if comments_fetched >= max_per_video:
                        break
                    comment = item['snippet']['topLevelComment']['snippet']
                    all_comments.append({
                        'video_id': video_id,
                        'comment_text': comment['textDisplay'],
                        'published_at': comment['publishedAt'],
                        'like_count': comment['likeCount']
                    })
                    comments_fetched += 1
                if comments_fetched >= max_per_video:
                    break
                request = youtube_client.commentThreads().list_next(request, response)
        except Exception as e:
            # st.warning(f"抓取影片 ID '{video_id}' 的留言時發生錯誤 (可能已關閉留言): {e}")
            continue
    return pd.DataFrame(all_comments)

# ========== 3. DeepSeek AI Sentiment Analysis (MODIFIED FOR ASYNC) ==========
async def analyze_comment_deepseek_async(comment_text, deepseek_client, semaphore, max_retries=3): # <<< MODIFIED
    import json
    if not isinstance(comment_text, str) or len(comment_text.strip()) < 5:
        return {"sentiment": "Invalid", "topic": "N/A", "summary": "Comment too short or invalid."}
    
    system_prompt = (
        "You are a professional Hong Kong market sentiment analyst. "
        "Analyze the following movie comment and strictly return the result in JSON format. "
        "The JSON object must contain three keys: "
        "1. 'sentiment': Must be either 'Positive', 'Negative', or 'Neutral'. "
        "2. 'topic': The core topic of the comment, e.g., 'Plot', 'Acting', 'Action Design', "
        "'Visuals', 'Pace', or 'Overall'. If unable to determine, use 'N/A'. "
        "3. 'summary': A concise one-sentence summary of the comment's main point. "
        "Ensure the output is only the JSON object and nothing else."
    )
    
    async with semaphore: # <<< MODIFIED: 控制並發數量
        for attempt in range(max_retries):
            try:
                # <<< MODIFIED: 使用 await 和異步 client
                response = await deepseek_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": comment_text}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                )
                analysis_result = json.loads(response.choices[0].message.content)
                return analysis_result
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt) # <<< MODIFIED: 使用 asyncio.sleep
                else:
                    return {"sentiment": "Error", "topic": "Error", "summary": f"API Error: {e}"}

# ========== 4. Main Process (MODIFIED FOR ASYNC) ==========
async def run_all_analyses(df, deepseek_client): # <<< MODIFIED: 新增的異步任務統籌函數
    # 控制並發數量，避免瞬間請求過多導致 API 拒絕。可根據 API rate limit 調整。
    semaphore = asyncio.Semaphore(50) 
    
    tasks = []
    for comment_text in df['comment_text']:
        tasks.append(analyze_comment_deepseek_async(comment_text, deepseek_client, semaphore))
        
    # 使用 tqdm 來顯示進度條
    from tqdm.asyncio import tqdm_asyncio
    results = await tqdm_asyncio.gather(*tasks, desc="AI Sentiment Analysis (Concurrent)")
    return results

def movie_comment_analysis(
    movie_title, start_date, end_date,
    yt_api_key, deepseek_api_key,
    max_videos_per_keyword=30, max_comments_per_video=50, sample_size=None
):
    # Keywords
    SEARCH_KEYWORDS = [
        f'"{movie_title}" 預告', f'"{movie_title}" review', f'"{movie_title}" 影評',
        f'"{movie_title}" 分析', f'"{movie_title}" 好唔好睇', f'"{movie_title}" 討論',
        f'"{movie_title}" reaction'
    ]
    # API init
    from googleapiclient.discovery import build
    youtube_client = build('youtube', 'v3', developerKey=yt_api_key)
    
    # <<< MODIFIED: 初始化 Async Client
    deepseek_client = openai.AsyncOpenAI(
        api_key=deepseek_api_key,
        base_url="https://api.deepseek.com/v1"
    )

    # Search videos
    video_ids = search_youtube_videos(
        SEARCH_KEYWORDS, youtube_client, max_videos_per_keyword, start_date, end_date
    )
    if not video_ids:
        return None, "找不到相關影片。"
    # Fetch comments
    df_comments = get_all_comments(video_ids, youtube_client, max_comments_per_video)
    if df_comments.empty:
        return None, "找不到任何留言。"
    # Time processing
    df_comments['published_at'] = pd.to_datetime(df_comments['published_at'], utc=True)
    df_comments['published_at_hk'] = df_comments['published_at'].dt.tz_convert('Asia/Hong_Kong')
    # Filter by HK timezone
    start = pd.to_datetime(start_date).tz_localize('Asia/Hong_Kong')
    end = pd.to_datetime(end_date).tz_localize('Asia/Hong_Kong') + timedelta(days=1)
    mask = (df_comments['published_at_hk'] >= start) & (df_comments['published_at_hk'] <= end)
    df_comments = df_comments.loc[mask].reset_index(drop=True)
    if df_comments.empty:
        return None, "在指定日期範圍內沒有留言。"
    # Sampling
    if sample_size and sample_size > 0 and sample_size < len(df_comments):
        df_analyze = df_comments.sample(n=sample_size, random_state=42)
    else:
        df_analyze = df_comments

    # <<< MODIFIED: 執行異步分析
    # 移除 tqdm.pandas，因為我們用自己的異步進度條
    st.info(f"準備對 {len(df_analyze)} 則留言進行高速並發分析...")
    analysis_results = asyncio.run(run_all_analyses(df_analyze, deepseek_client))
    
    analysis_df = pd.json_normalize(analysis_results)
    final_df = pd.concat([df_analyze.reset_index(drop=True), analysis_df], axis=1)
    final_df['published_at'] = pd.to_datetime(final_df['published_at'])
    return final_df, None

# ========== 5. Streamlit UI (MODIFIED FOR PROGRESS BAR) ==========
st.set_page_config(page_title="YouTube 電影評論 AI 分析", layout="wide")
st.title("🎬 YouTube 電影評論 AI 情感分析")

with st.expander("使用說明"):
    st.markdown("""
    1.  輸入電影的**中文全名**、分析時間範圍及所需的 API 金鑰。
    2.  自訂每個關鍵字搜尋的影片數量上限，及每部影片抓取的留言數量上限。
    3.  點擊「開始分析」，系統將自動抓取 YouTube 留言並進行 AI 高速情感分析。
    4.  分析完成後，下方會顯示數據圖表及詳細結果的下載按鈕。
    """)

movie_title = st.text_input("電影名稱 (建議使用香港通用的中文全名)", value="九龍城寨之圍城")
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("開始日期", value=datetime.today() - timedelta(days=30))
with col2:
    end_date = st.date_input("結束日期", value=datetime.today())
yt_api_key = st.text_input("YouTube API Key", type='password')
deepseek_api_key = st.text_input("DeepSeek API Key", type='password')

st.subheader("進階設定")
max_videos = st.slider("每個關鍵字的最大影片搜尋數", 5, 50, 10, help="增加此數值會找到更多影片，但會增加 YouTube API 的配額消耗。")
max_comments = st.slider("每部影片的最大留言抓取數", 10, 200, 50, help="分析的主要來源，數量越多，分析結果越全面，但 DeepSeek API 成本越高。")
sample_size = st.number_input("分析留言數量上限 (0 代表分析全部已抓取的留言)", 0, 5000, 500, help="設定一個上限以控制分析時間和成本。例如，即使抓取了 2000 則留言，這裡設 500 就只會分析其中的 500 則。")

if st.button("🚀 開始分析"):
    if not all([movie_title, yt_api_key, deepseek_api_key]):
        st.warning("請填寫電影名稱和兩個 API 金鑰。")
    else:
        # <<< MODIFIED: 移除 with st.spinner，因為我們在函數內有自己的進度提示
        # 我們需要一個 placeholder 來顯示 tqdm 的進度條
        progress_placeholder = st.empty()
        
        # 由於 Streamlit 的限制，直接在 UI 顯示 tqdm 進度條比較困難
        # 我們改為在後端打印，並在前端顯示一個通用的 spinner
        with st.spinner("AI 高速分析中... (處理 500 則留言約需 1-2 分鐘)"):
            df_result, err = movie_comment_analysis(
                movie_title, str(start_date), str(end_date),
                yt_api_key, deepseek_api_key,
                max_videos, max_comments, sample_size
            )
        
        if err:
            st.error(err)
        else:
            st.success("分析完成！")
            st.dataframe(df_result.head(20))

            # ========== 可视化 (No changes) ==========
            st.subheader("1. 情感分佈圓餅圖")
            fig1, ax1 = plt.subplots(figsize=(5, 4))
            valmap = {
                "Positive": "正面", "Negative": "負面", "Neutral": "中性",
                "Invalid": "無效", "Error": "錯誤"
            }
            # 確保所有可能的值都存在，避免 Key-Error
            df_result['sentiment_cn'] = df_result['sentiment'].map(lambda x: valmap.get(str(x), str(x)))
            
            # 定義顏色和順序，確保圖表一致性
            s_counts = df_result['sentiment_cn'].value_counts()
            order = ['正面', '負面', '中性', '無效', '錯誤']
            colors_map = {'正面': '#5cb85c', '負面': '#d9534f', '中性': '#f0ad4e', '無效': '#cccccc', '錯誤': '#888888'}
            
            # 過濾掉不存在的標籤
            s_counts = s_counts.reindex(order).dropna()
            
            s_counts.plot.pie(
                autopct='%.1f%%', ax=ax1,
                colors=[colors_map[key] for key in s_counts.index],
                wedgeprops={'linewidth': 1.0, 'edgecolor': 'white'}
            )
            ax1.set_title('整體情感分佈', fontsize=16)
            ax1.set_ylabel('')
            st.pyplot(fig1, use_container_width=False)

            st.subheader("2. 每日情感趨勢 (堆疊長條圖)")
            df_result['date'] = df_result['published_at_hk'].dt.date
            daily = df_result.groupby(['date', 'sentiment_cn']).size().unstack().fillna(0)
            
            # 確保欄位順序
            daily = daily.reindex(columns=order).dropna(axis=1, how='all')

            fig2, ax2 = plt.subplots(figsize=(10, 4))
            daily.plot(kind='bar', stacked=True, ax=ax2, width=0.8, 
                       color=[colors_map[col] for col in daily.columns])
            ax2.set_title('每日情感趨勢', fontsize=16)
            ax2.set_xlabel('日期')
            ax2.set_ylabel('留言數量')
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='情感')
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)

            st.subheader("3. 各主題情感佔比")
            topic_sentiment = df_result.groupby(['topic', 'sentiment_cn']).size().unstack().fillna(0)
            topic_sentiment = topic_sentiment.reindex(columns=order).dropna(axis=1, how='all')
            
            # 計算百分比
            topic_sentiment_percent = topic_sentiment.div(topic_sentiment.sum(axis=1), axis=0) * 100

            fig3, ax3 = plt.subplots(figsize=(10, 5))
            topic_sentiment_percent.plot(kind='bar', stacked=True, ax=ax3,
                                         color=[colors_map[col] for col in topic_sentiment_percent.columns])
            ax3.set_title('各討論主題的情感佔比', fontsize=16)
            ax3.set_xlabel('主題')
            ax3.set_ylabel('百分比 (%)')
            ax3.yaxis.set_major_formatter(plt.FuncFormatter('{:.0f}%'.format))
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='情感')
            plt.tight_layout()
            st.pyplot(fig3, use_container_width=True)

            st.subheader("4. 下載分析明細")
            csv = df_result.to_csv(index=False, encoding='utf-8-sig')
            st.download_button("📥 下載全部分析明細 (CSV)", csv, file_name=f"{movie_title}_analysis_details.csv", mime='text/csv')
