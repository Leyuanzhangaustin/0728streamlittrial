# app.py (Accelerated Version with English Visualizations)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
            st.warning(f"Error searching for keyword '{query}': {e}")
            continue
    return list(all_video_ids)

# ========== 2. Batch Fetch Comments (No changes) ==========
def get_all_comments(video_ids, youtube_client, max_per_video):
    all_comments = []
    for video_id in video_ids:
        try:
            # Request comments sorted by 'time' (newest first)
            request = youtube_client.commentThreads().list(
                part='snippet',
                videoId=video_id,
                textFormat='plainText',
                maxResults=100,
                order='time'  # <<< CRITICAL CHANGE: Sort by newest first
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
                # Check if there is a next page
                if 'nextPageToken' in response:
                    request = youtube_client.commentThreads().list_next(request, response)
                else:
                    request = None # No more pages
        except Exception as e:
            # Silently continue if comments are disabled or other errors occur
            continue
    return pd.DataFrame(all_comments)

# ========== 3. DeepSeek AI Sentiment Analysis (Async) (No changes) ==========
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

# ========== 4. Main Process (Async) (No changes) ==========
async def run_all_analyses(df, deepseek_client):
    semaphore = asyncio.Semaphore(50) 
    tasks = []
    for comment_text in df['comment_text']:
        tasks.append(analyze_comment_deepseek_async(comment_text, deepseek_client, semaphore))
        
    from tqdm.asyncio import tqdm_asyncio
    results = await tqdm_asyncio.gather(*tasks, desc="AI Sentiment Analysis (Concurrent)")
    return results

def movie_comment_analysis(
    movie_title, start_date, end_date,
    yt_api_key, deepseek_api_key,
    max_videos_per_keyword=30, max_comments_per_video=50, sample_size=None
):
    SEARCH_KEYWORDS = [
        f'"{movie_title}" trailer', f'"{movie_title}" review', f'"{movie_title}" official',
        f'"{movie_title}" analysis', f'"{movie_title}" discussion', f'"{movie_title}" reaction'
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
        return None, "No relevant videos found."
    df_comments = get_all_comments(video_ids, youtube_client, max_comments_per_video)
    if df_comments.empty:
        return None, "No comments found for the retrieved videos."
    df_comments['published_at'] = pd.to_datetime(df_comments['published_at'], utc=True)
    df_comments['published_at_hk'] = df_comments['published_at'].dt.tz_convert('Asia/Hong_Kong')
    start = pd.to_datetime(start_date).tz_localize('Asia/Hong_Kong')
    end = pd.to_datetime(end_date).tz_localize('Asia/Hong_Kong') + timedelta(days=1)
    mask = (df_comments['published_at_hk'] >= start) & (df_comments['published_at_hk'] <= end)
    df_comments = df_comments.loc[mask].reset_index(drop=True)
    if df_comments.empty:
        return None, "No comments found within the specified date range."
    if sample_size and sample_size > 0 and sample_size < len(df_comments):
        df_analyze = df_comments.sample(n=sample_size, random_state=42)
    else:
        df_analyze = df_comments

    st.info(f"Starting high-speed concurrent analysis for {len(df_analyze)} comments...")
    analysis_results = asyncio.run(run_all_analyses(df_analyze, deepseek_client))
    
    analysis_df = pd.json_normalize(analysis_results)
    final_df = pd.concat([df_analyze.reset_index(drop=True), analysis_df], axis=1)
    final_df['published_at'] = pd.to_datetime(final_df['published_at'])
    return final_df, None

# ========== 5. Streamlit UI & Visualization (MODIFIED) ==========
st.set_page_config(page_title="YouTube Movie Comment AI Analysis", layout="wide")
st.title("🎬 YouTube Movie Comment AI Sentiment Analysis")

with st.expander("Instructions"):
    st.markdown("""
    1.  Enter the movie title, analysis date range, and the required API keys.
    2.  Customize the maximum number of videos to search per keyword and comments to fetch per video.
    3.  Click "Start Analysis" to fetch comments and run high-speed AI sentiment analysis.
    4.  After completion, view the data visualizations and download the detailed results.
    """)

movie_title = st.text_input("Movie Title", value="Twilight of the Warriors: Walled In")
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=datetime.today() - timedelta(days=30))
with col2:
    end_date = st.date_input("End Date", value=datetime.today())
yt_api_key = st.text_input("YouTube API Key", type='password')
deepseek_api_key = st.text_input("DeepSeek API Key", type='password')

st.subheader("Advanced Settings")
max_videos = st.slider("Max Videos per Keyword", 5, 50, 10, help="Increasing this finds more videos but consumes more YouTube API quota.")
max_comments = st.slider("Max Comments per Video", 10, 200, 50, help="More comments provide a more comprehensive analysis but increase DeepSeek API costs.")
sample_size = st.number_input("Max Comments to Analyze (0 = analyze all)", 0, 5000, 500, help="Set a limit to control analysis time and cost. E.g., if 2000 comments are fetched, setting this to 500 will only analyze a random sample of 500.")

if st.button("🚀 Start Analysis"):
    if not all([movie_title, yt_api_key, deepseek_api_key]):
        st.warning("Please provide the Movie Title and both API Keys.")
    else:
        with st.spinner("AI is analyzing at high speed... (Analyzing 500 comments takes about 1-2 minutes)"):
            df_result, err = movie_comment_analysis(
                movie_title, str(start_date), str(end_date),
                yt_api_key, deepseek_api_key,
                max_videos, max_comments, sample_size
            )
        
        if err:
            st.error(err)
        else:
            st.success("Analysis Complete!")
            st.dataframe(df_result.head(20))

            # ========== Visualization (MODIFIED FOR ENGLISH & LINE CHART) ==========
            
            # --- Data Preparation for Visualization ---
            # <<< MODIFIED: Use English labels for consistency
            valmap = {
                "Positive": "Positive", "Negative": "Negative", "Neutral": "Neutral",
                "Invalid": "Invalid", "Error": "Error"
            }
            df_result['sentiment_en'] = df_result['sentiment'].map(lambda x: valmap.get(str(x), str(x)))
            
            order = ['Positive', 'Negative', 'Neutral', 'Invalid', 'Error']
            colors_map = {'Positive': '#5cb85c', 'Negative': '#d9534f', 'Neutral': '#f0ad4e', 'Invalid': '#cccccc', 'Error': '#888888'}

            # --- Chart 1: Pie Chart ---
            st.subheader("1. Overall Sentiment Distribution")
            fig1, ax1 = plt.subplots(figsize=(5, 4))
            
            s_counts = df_result['sentiment_en'].value_counts()
            s_counts = s_counts.reindex(order).dropna()
            
            s_counts.plot.pie(
                autopct='%.1f%%', ax=ax1,
                colors=[colors_map[key] for key in s_counts.index],
                wedgeprops={'linewidth': 1.0, 'edgecolor': 'white'}
            )
            ax1.set_title('Overall Sentiment Distribution', fontsize=16)
            ax1.set_ylabel('')
            st.pyplot(fig1, use_container_width=False)

            # --- Data Preparation for Daily Trend ---
            df_result['date'] = df_result['published_at_hk'].dt.date
            daily = df_result.groupby(['date', 'sentiment_en']).size().unstack().fillna(0)
            daily = daily.reindex(columns=order).dropna(axis=1, how='all')

            # --- Chart 2: Stacked Bar Chart ---
            st.subheader("2. Daily Sentiment Volume (Stacked Bar Chart)")
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            daily.plot(kind='bar', stacked=True, ax=ax2, width=0.8, 
                       color=[colors_map.get(col) for col in daily.columns])
            ax2.set_title('Daily Comment Volume by Sentiment', fontsize=16)
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Number of Comments')
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Sentiment')
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)

            # --- Chart 3: Line Chart ---
            # <<< NEW: Added line chart for trend comparison
            st.subheader("3. Daily Sentiment Trend (Line Chart)")
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            
            # Plot only the main sentiments for clarity
            for sentiment in ['Positive', 'Negative', 'Neutral']:
                if sentiment in daily.columns:
                    ax3.plot(daily.index, daily[sentiment], marker='o', linestyle='-', label=sentiment, color=colors_map[sentiment])
            
            ax3.set_title('Daily Sentiment Trends', fontsize=16)
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Number of Comments')
            ax3.legend(title='Sentiment')
            ax3.grid(axis='y', linestyle='--', alpha=0.7)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig3, use_container_width=True)

            # --- Chart 4: Topic Analysis ---
            st.subheader("4. Sentiment Distribution by Topic")
            # Filter out invalid/error topics for a cleaner chart
            topic_df = df_result[~df_result['topic'].isin(['N/A', 'Error'])]
            topic_sentiment = topic_df.groupby(['topic', 'sentiment_en']).size().unstack().fillna(0)
            topic_sentiment = topic_sentiment.reindex(columns=order).dropna(axis=1, how='all')
            
            if not topic_sentiment.empty:
                topic_sentiment_percent = topic_sentiment.div(topic_sentiment.sum(axis=1), axis=0) * 100
                fig4, ax4 = plt.subplots(figsize=(10, 5))
                topic_sentiment_percent.plot(kind='bar', stacked=True, ax=ax4,
                                             color=[colors_map.get(col) for col in topic_sentiment_percent.columns])
                ax4.set_title('Sentiment Breakdown by Topic', fontsize=16)
                ax4.set_xlabel('Topic')
                ax4.set_ylabel('Percentage (%)')
                ax4.yaxis.set_major_formatter(plt.FuncFormatter('{:.0f}%'.format))
                plt.xticks(rotation=45, ha='right')
                plt.legend(title='Sentiment')
                plt.tight_layout()
                st.pyplot(fig4, use_container_width=True)
            else:
                st.info("Not enough data with specific topics to generate a topic analysis chart.")


            # --- Chart 5: Download Button ---
            st.subheader("5. Download Full Analysis")
            csv = df_result.to_csv(index=False, encoding='utf-8-sig')
            st.download_button("📥 Download Full Analysis Details (CSV)", csv, file_name=f"{movie_title}_analysis_details.csv", mime='text/csv')
