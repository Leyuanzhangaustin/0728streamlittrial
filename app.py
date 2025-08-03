# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time

# ========== 1. YouTube Search ==========
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
            continue
    return list(all_video_ids)

# ========== 2. Batch Fetch Comments ==========
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
            continue
    return pd.DataFrame(all_comments)

# ========== 3. DeepSeek AI Sentiment Analysis ==========
def analyze_comment_deepseek(comment_text, deepseek_client, max_retries=3):
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
    for attempt in range(max_retries):
        try:
            response = deepseek_client.chat.completions.create(
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
                time.sleep(2 ** attempt)
            else:
                return {"sentiment": "Error", "topic": "Error", "summary": f"API Error: {e}"}

# ========== 4. Main Process ==========
def movie_comment_analysis(
    movie_title, start_date, end_date,
    yt_api_key, deepseek_api_key,
    max_videos_per_keyword=30, max_comments_per_video=50, sample_size=None
):
    # Keywords
    SEARCH_KEYWORDS = [
        f'"{movie_title}" trailer',
        f'"{movie_title}" review',
        f'"{movie_title}" analysis',
        f'"{movie_title}" good or not',
        f'"{movie_title}" discussion',
        f'"{movie_title}" reaction'
    ]
    # API init
    from googleapiclient.discovery import build
    youtube_client = build('youtube', 'v3', developerKey=yt_api_key)
    import openai
    deepseek_client = openai.OpenAI(
        api_key=deepseek_api_key,
        base_url="https://api.deepseek.com/v1"
    )

    # Search videos
    video_ids = search_youtube_videos(
        SEARCH_KEYWORDS, youtube_client, max_videos_per_keyword, start_date, end_date
    )
    if not video_ids:
        return None, "No relevant videos found."
    # Fetch comments
    df_comments = get_all_comments(video_ids, youtube_client, max_comments_per_video)
    if df_comments.empty:
        return None, "No comments found."
    # Time processing
    df_comments['published_at'] = pd.to_datetime(df_comments['published_at'], utc=True)
    df_comments['published_at_hk'] = df_comments['published_at'].dt.tz_convert('Asia/Hong_Kong')
    # Filter by HK timezone
    start = pd.to_datetime(start_date).tz_localize('Asia/Hong_Kong')
    end = pd.to_datetime(end_date).tz_localize('Asia/Hong_Kong') + timedelta(days=1)
    mask = (df_comments['published_at_hk'] >= start) & (df_comments['published_at_hk'] <= end)
    df_comments = df_comments.loc[mask].reset_index(drop=True)
    if df_comments.empty:
        return None, "No comments within date range."
    # Sampling
    if sample_size and sample_size < len(df_comments):
        df_analyze = df_comments.sample(n=sample_size, random_state=42)
    else:
        df_analyze = df_comments

    # AI Sentiment Analysis
    from tqdm import tqdm
    tqdm.pandas(desc="AI Sentiment Analysis")
    analysis_results = df_analyze['comment_text'].progress_apply(
        lambda x: analyze_comment_deepseek(x, deepseek_client)
    )
    analysis_df = pd.json_normalize(analysis_results)
    final_df = pd.concat([df_analyze.reset_index(drop=True), analysis_df], axis=1)
    final_df['published_at'] = pd.to_datetime(final_df['published_at'])
    return final_df, None

# ========== 5. Streamlit UI ==========
st.set_page_config(page_title="YouTube Movie Comment AI Analysis", layout="wide")
st.title("🎬 YouTube Movie Comment AI Sentiment Analysis")

with st.expander("Instructions"):
    st.markdown("""
    1. Enter the movie title, analysis time range, and API keys.
    2. Customize the max number of videos per keyword and comments per video.
    3. Click "Start Analysis" to fetch and analyze comments.
    4. After analysis, view visualizations and download details.
    """)

movie_title = st.text_input("Movie Title", value="")
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=datetime.today() - timedelta(days=30))
with col2:
    end_date = st.date_input("End Date", value=datetime.today())
yt_api_key = st.text_input("YouTube API Key", type='password')
deepseek_api_key = st.text_input("DeepSeek API Key", type='password')
max_videos = st.slider("Max Videos per Keyword", 5, 50, 10)
max_comments = st.slider("Max Comments per Video", 5, 100, 20)
sample_size = st.number_input("Max Number of Comments to Analyze (0 = all)", 0, 2000, 0)

if st.button("🚀 Start Analysis"):
    if not all([movie_title, yt_api_key, deepseek_api_key]):
        st.warning("Please fill in all fields.")
    else:
        with st.spinner("AI analyzing... (may take a few minutes for many comments)"):
            df_result, err = movie_comment_analysis(
                movie_title, str(start_date), str(end_date),
                yt_api_key, deepseek_api_key,
                max_videos, max_comments, sample_size or None
            )
        if err:
            st.error(err)
        else:
            st.success("Analysis complete!")
            st.dataframe(df_result.head(20))

 # ========== 可视化 ==========
            st.subheader("1. Sentiment Distribution")
            fig1, ax1 = plt.subplots(figsize=(4, 4))  # Smaller figure
            valmap = {
                "Positive": "Positive",
                "Negative": "Negative",
                "Neutral": "Neutral",
                "Invalid": "Invalid",
                "Error": "Error"
            }
            df_result['sentiment_en'] = df_result['sentiment'].map(lambda x: valmap.get(str(x).capitalize(), x))
            df_result['sentiment_en'].value_counts().plot.pie(
                autopct='%.1f%%',
                ax=ax1,
                colors=['#5cb85c', '#d9534f', '#f0ad4e', '#cccccc', '#888888']
            )
            ax1.set_title('Sentiment Distribution')
            ax1.set_ylabel('')
            st.pyplot(fig1, use_container_width=False)

            st.subheader("2. Daily Sentiment Trend (Bar)")
            df_result['date'] = df_result['published_at_hk'].dt.date
            daily = df_result.groupby(['date', 'sentiment_en']).size().unstack().fillna(0)
            fig2, ax2 = plt.subplots(figsize=(8, 3))
            daily.plot(kind='bar', stacked=True, ax=ax2, width=0.8, color=['#5cb85c', '#d9534f', '#f0ad4e', '#cccccc', '#888888'])
            ax2.set_title('Daily Sentiment Trend (Bar)')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Number of Comments')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)

            st.subheader("3. Daily Sentiment Trend (Line)")
            fig3, ax3 = plt.subplots(figsize=(8, 3))
            for sentiment in ['Positive', 'Negative', 'Neutral']:
                if sentiment in daily.columns:
                    ax3.plot(daily.index, daily[sentiment], marker='o', label=sentiment)
            ax3.set_title("Daily Sentiment Trend (Line)")
            ax3.set_xlabel("Date")
            ax3.set_ylabel("Number of Comments")
            ax3.legend(title="Sentiment")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig3, use_container_width=True)

            st.subheader("4. 下载分析明细")
            csv = df_result.to_csv(index=False, encoding='utf-8-sig')
            st.download_button("下载全部分析明细CSV", csv, file_name=f"{movie_title}_analysis.csv", mime='text/csv')
