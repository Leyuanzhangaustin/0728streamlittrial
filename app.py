# app.py (Final Visualization Version)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.express as px
from datetime import datetime, timedelta
import time
import asyncio
import openai

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
            time.sleep(0.5)
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
        except Exception:
            continue
    return pd.DataFrame(all_comments)

# ========== 3. DeepSeek AI Sentiment Analysis (No changes) ==========
async def analyze_comment_deepseek_async(comment_text, deepseek_client, semaphore, max_retries=3):
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

    async with semaphore:
        for attempt in range(max_retries):
            try:
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
                    await asyncio.sleep(2 ** attempt)
                else:
                    return {"sentiment": "Error", "topic": "Error", "summary": f"API Error: {e}"}

# ========== 4. Main Process (No changes) ==========
async def run_all_analyses(df, deepseek_client):
    semaphore = asyncio.Semaphore(50)
    tasks = [
        analyze_comment_deepseek_async(comment_text, deepseek_client, semaphore)
        for comment_text in df['comment_text']
    ]

    from tqdm.asyncio import tqdm_asyncio
    results = await tqdm_asyncio.gather(*tasks, desc="AI Sentiment Analysis (Concurrent)")
    return results

def movie_comment_analysis(
    movie_title, start_date, end_date,
    yt_api_key, deepseek_api_key,
    max_videos_per_keyword=30, max_comments_per_video=50, sample_size=None
):
    SEARCH_KEYWORDS = [
        f'"{movie_title}" 預告', f'"{movie_title}" review', f'"{movie_title}" 影評',
        f'"{movie_title}" 分析', f'"{movie_title}" 好唔好睇', f'"{movie_title}" 討論',
        f'"{movie_title}" reaction'
    ]

    from googleapiclient.discovery import build
    youtube_client = build('youtube', 'v3', developerKey=yt_api_key)

    deepseek_client = openai.AsyncOpenAI(
        api_key=deepseek_api_key,
        base_url="https://api.deepseek.com/v1"
    )

    video_ids = search_youtube_videos(
        SEARCH_KEYWORDS, youtube_client, max_videos_per_keyword, start_date, end_date
    )
    if not video_ids:
        return None, "找不到相關影片。"

    df_comments = get_all_comments(video_ids, youtube_client, max_comments_per_video)
    if df_comments.empty:
        return None, "找不到任何留言。"

    df_comments['published_at'] = pd.to_datetime(df_comments['published_at'], utc=True)
    df_comments['published_at_hk'] = df_comments['published_at'].dt.tz_convert('Asia/Hong_Kong')

    start = pd.to_datetime(start_date).tz_localize('Asia/Hong_Kong')
    end = pd.to_datetime(end_date).tz_localize('Asia/Hong_Kong') + timedelta(days=1)
    mask = (df_comments['published_at_hk'] >= start) & (df_comments['published_at_hk'] <= end)
    df_comments = df_comments.loc[mask].reset_index(drop=True)
    if df_comments.empty:
        return None, "在指定日期範圍內沒有留言。"

    if sample_size and sample_size > 0 and sample_size < len(df_comments):
        df_analyze = df_comments.sample(n=sample_size, random_state=42)
    else:
        df_analyze = df_comments

    st.info(f"準備對 {len(df_analyze)} 則留言進行高速並發分析...")
    analysis_results = asyncio.run(run_all_analyses(df_analyze, deepseek_client))

    analysis_df = pd.json_normalize(analysis_results)
    final_df = pd.concat([df_analyze.reset_index(drop=True), analysis_df], axis=1)
    final_df['published_at'] = pd.to_datetime(final_df['published_at'])
    return final_df, None

# ========== 5. Streamlit UI (No changes in this part) ==========
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
            
            st.header("📊 可視化分析結果")

            # --- 共用設定 ---
            sentiments_order = ['Positive', 'Negative', 'Neutral', 'Invalid', 'Error']
            colors_map = {
                'Positive': '#5cb85c', 'Negative': '#d9534f', 'Neutral': '#f0ad4e',
                'Invalid': '#cccccc', 'Error': '#888888'
            }

            # --- 1. 情感分佈圓餅圖 (No changes) ---
            st.subheader("1. Sentiment Distribution (Pie)")
            sentiment_series = df_result['sentiment'].dropna().astype(str)
            sentiment_counts = sentiment_series.value_counts()
            ordered_labels = [label for label in sentiments_order if label in sentiment_counts.index]

            if not sentiment_counts.empty:
                fig1, ax1 = plt.subplots(figsize=(5, 4))
                ax1.pie(
                    sentiment_counts[ordered_labels],
                    labels=ordered_labels,
                    autopct='%.1f%%',
                    colors=[colors_map[label] for label in ordered_labels],
                    wedgeprops={'linewidth': 1.0, 'edgecolor': 'white'}
                )
                ax1.set_title('Overall Sentiment Distribution', fontsize=16)
                st.pyplot(fig1, use_container_width=False)
            else:
                st.info("No sentiment data available for pie chart.")

            # <<< MODIFIED BLOCK START: 實現兩張獨立的每日趨勢圖 >>>
            
            st.subheader("2. Daily Sentiment Trend")
            
            # --- 數據準備 (共用) ---
            if 'published_at_hk' in df_result.columns:
                df_result['date'] = df_result['published_at_hk'].dt.date
            else:
                df_result['date'] = df_result['published_at'].dt.date
            
            daily = df_result.groupby(['date', 'sentiment']).size().unstack().fillna(0)
            daily = daily.reindex(columns=sentiments_order).dropna(axis=1, how='all')

            if not daily.empty:
                # 將數據從 "wide" 轉為 "long" 格式，方便 Plotly 使用
                daily_long = daily.reset_index().melt(id_vars='date', var_name='sentiment', value_name='count')
                
                # --- 圖表 2a: 每日情感趨勢 (折線圖) ---
                st.markdown("#### 每日情感趨勢 (折線圖)")
                st.markdown("此圖表展示各情感類別每日的留言數量變化，適合比較不同情感的熱度趨勢。")
                
                fig_line = px.line(
                    daily_long,
                    x='date',
                    y='count',
                    color='sentiment',
                    title='Daily Comment Volume Trend by Sentiment',
                    labels={'date': 'Date', 'count': 'Number of Comments', 'sentiment': 'Sentiment'},
                    color_discrete_map=colors_map,
                    category_orders={'sentiment': [col for col in sentiments_order if col in daily.columns]}
                )
                fig_line.update_layout(legend_title_text='Sentiment')
                st.plotly_chart(fig_line, use_container_width=True)

                # --- 圖表 2b: 每日留言總量 (堆疊長條圖) ---
                st.markdown("#### 每日留言總量及情感分佈 (堆疊長條圖)")
                st.markdown("此圖表展示每日的總留言量，並以顏色區分其中各種情感的佔比。")

                fig_bar = px.bar(
                    daily_long,
                    x='date',
                    y='count',
                    color='sentiment',
                    title='Daily Comment Volume by Sentiment (Stacked)',
                    labels={'date': 'Date', 'count': 'Number of Comments', 'sentiment': 'Sentiment'},
                    color_discrete_map=colors_map,
                    category_orders={'sentiment': [col for col in sentiments_order if col in daily.columns]}
                )
                fig_bar.update_layout(legend_title_text='Sentiment', barmode='stack')
                st.plotly_chart(fig_bar, use_container_width=True)

            else:
                st.info("Not enough daily sentiment data to display the trend charts.")

            # <<< MODIFIED BLOCK END >>>

            # --- 3. 各主題情感佔比 (No changes) ---
            st.subheader("3. Sentiment Share by Topic")
            topic_sentiment = df_result.groupby(['topic', 'sentiment']).size().unstack().fillna(0)
            topic_sentiment = topic_sentiment.reindex(columns=sentiments_order).dropna(axis=1, how='all')
            
            if not topic_sentiment.empty:
                topic_sentiment = topic_sentiment[topic_sentiment.sum(axis=1) > 0]
                
                if not topic_sentiment.empty:
                    topic_sentiment_percent = topic_sentiment.div(topic_sentiment.sum(axis=1), axis=0).fillna(0) * 100

                    fig3, ax3 = plt.subplots(figsize=(10, 5))
                    topic_sentiment_percent.plot(
                        kind='bar',
                        stacked=True,
                        ax=ax3,
                        color=[colors_map[col] for col in topic_sentiment_percent.columns]
                    )
                    ax3.set_title('Sentiment Share by Topic', fontsize=16)
                    ax3.set_xlabel('Topic')
                    ax3.set_ylabel('Percentage (%)')
                    ax3.yaxis.set_major_formatter(plt.FuncFormatter('{:.0f}%'.format))
                    plt.xticks(rotation=45, ha='right')
                    ax3.legend(title='Sentiment')
                    plt.tight_layout()
                    st.pyplot(fig3, use_container_width=True)
                else:
                    st.info("No topic data with comments to display the chart.")
            else:
                st.info("Not enough topic sentiment data to display the stacked bar chart.")

            # --- 4. 下載分析明細 (No changes) ---
            st.subheader("4. 下載分析明細")
            csv = df_result.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                "📥 下載全部分析明細 (CSV)",
                csv,
                file_name=f"{movie_title}_analysis_details.csv",
                mime='text/csv'
            )
