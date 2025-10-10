import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
            time.sleep(0.5)  # Keep a small delay to be nice to YouTube API
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

# ========== 3. DeepSeek AI Sentiment Analysis (Async) ==========
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

# ========== 4. Main Process (Async Coordinator) ==========
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

# ========== 5. Streamlit UI ==========
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
        progress_placeholder = st.empty()

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

            st.subheader("1. 情感分佈圓餅圖")
            fig1, ax1 = plt.subplots(figsize=(5, 4))
            valmap = {
                "Positive": "正面", "Negative": "負面", "Neutral": "中性",
                "Invalid": "無效", "Error": "錯誤"
            }
            df_result['sentiment_cn'] = df_result['sentiment'].map(lambda x: valmap.get(str(x), str(x)))
            s_counts = df_result['sentiment_cn'].value_counts()
            order = ['正面', '負面', '中性', '無效', '錯誤']
            colors_map = {'正面': '#5cb85c', '負面': '#d9534f', '中性': '#f0ad4e', '無效': '#cccccc', '錯誤': '#888888'}
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
            df_result['date'] = df_result['published_at_hk'].dt.floor('D')
            daily = df_result.groupby(['date', 'sentiment_cn']).size().unstack().fillna(0)
            daily = daily.reindex(columns=order).dropna(axis=1, how='all')
            daily = daily.sort_index()
            daily.index = pd.to_datetime(daily.index)

            fig2, ax2 = plt.subplots(figsize=(12, 4))

            daily_plot_df = daily.reset_index()
            dates = daily_plot_df['date'].to_list()
            bottom = np.zeros(len(dates))

            for sentiment in daily.columns:
                values = daily_plot_df[sentiment].to_numpy()
                if np.any(values):
                    ax2.bar(
                        dates,
                        values,
                        bottom=bottom,
                        width=0.8,
                        color=colors_map.get(sentiment, '#999999'),
                        label=sentiment
                    )
                    bottom += values

            ax2.set_title('每日情感趨勢', fontsize=16)
            ax2.set_xlabel('日期')
            ax2.set_ylabel('留言數量')

            num_dates = len(dates)
            if num_dates > 0:
                if num_dates > 20:
                    interval = max(5, int(np.ceil(num_dates / 12)))
                else:
                    interval = 1

                locator = mdates.DayLocator(interval=interval)
                formatter = mdates.DateFormatter('%Y-%m-%d')
                ax2.xaxis.set_major_locator(locator)
                ax2.xaxis.set_major_formatter(formatter)

                # 確保最後一天一定出現在刻度上
                if num_dates > 1:
                    last_date = dates[-1]
                    tick_dates = locator.tick_values(dates[0], dates[-1])
                    if last_date not in pd.to_datetime(tick_dates):
                        ax2.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=13))
                        ax2.xaxis.set_major_formatter(formatter)

                plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=9)
                ax2.set_xlim(dates[0] - pd.Timedelta(days=0.5), dates[-1] + pd.Timedelta(days=0.5))
            else:
                ax2.set_xticks([])

            ax2.legend(title='情感')
            fig2.subplots_adjust(bottom=0.28)
            st.pyplot(fig2, use_container_width=True)

            st.subheader("3. 各主題情感佔比")
            topic_sentiment = df_result.groupby(['topic', 'sentiment_cn']).size().unstack().fillna(0)
            topic_sentiment = topic_sentiment.reindex(columns=order).dropna(axis=1, how='all')
            topic_sentiment_percent = topic_sentiment.div(topic_sentiment.sum(axis=1), axis=0) * 100

            fig3, ax3 = plt.subplots(figsize=(10, 5))
            topic_sentiment_percent.plot(
                kind='bar',
                stacked=True,
                ax=ax3,
                color=[colors_map[col] for col in topic_sentiment_percent.columns]
            )
            ax3.set_title('各討論主題的情感佔比', fontsize=16)
            ax3.set_xlabel('主題')
            ax3.set_ylabel('百分比 (%)')
            ax3.yaxis.set_major_formatter(plt.FuncFormatter('{:.0f}%'.format))
            plt.xticks(rotation=45, ha='right')
            ax3.legend(title='情感')
            plt.tight_layout()
            st.pyplot(fig3, use_container_width=True)

            st.subheader("4. 下載分析明細")
            csv = df_result.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                "📥 下載全部分析明細 (CSV)",
                csv,
                file_name=f"{movie_title}_analysis_details.csv",
                mime='text/csv'
