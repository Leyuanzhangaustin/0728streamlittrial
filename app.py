import streamlit as st
import pandas as pd
from googleapiclient.discovery import build
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re
import html
from datetime import datetime

# ==========================================
# 1. Setup & Helper Functions
# ==========================================

st.set_page_config(page_title="YouTube Movie Sentiment Analysis", layout="wide")

# ---------------------------------------------------------
# NATIVE SENTIMENT DICTIONARY (No external libraries needed)
# ---------------------------------------------------------
POSITIVE_WORDS = {
    'good', 'great', 'excellent', 'amazing', 'love', 'best', 'awesome', 
    'fantastic', 'happy', 'enjoy', 'beautiful', 'nice', 'perfect', 'fun', 
    'funny', 'interesting', 'brilliant', 'wonderful', 'super', 'cool', 
    'entertaining', 'masterpiece', 'loved', 'likes', 'excited', 'incredible',
    'touching', 'emotional', 'epic', 'legendary', 'classic', 'solid', 'recommend'
}

NEGATIVE_WORDS = {
    'bad', 'terrible', 'awful', 'worst', 'hate', 'boring', 'disappointing', 
    'poor', 'horrible', 'waste', 'stupid', 'annoying', 'sad', 'disaster', 
    'fail', 'ugly', 'mess', 'trash', 'useless', 'dumb', 'cringe', 'weak',
    'confusing', 'slow', 'dry', 'predictable', 'cliche', 'unfunny', 'garbage',
    'ruined', 'cheap', 'lazy', 'pointless'
}

STOPWORDS = {
    'the', 'a', 'an', 'in', 'on', 'at', 'of', 'to', 'is', 'are', 'was', 'were',
    'and', 'but', 'or', 'so', 'it', 'this', 'that', 'my', 'your', 'his', 'her',
    'movie', 'film', 'video', 'really', 'very', 'just', 'like', 'watch', 
    'watching', 'time', 'people', 'think', 'know', 'would', 'could', 'should', 
    'get', 'got', 'make', 'made', 'see', 'saw', 'seen', 'one', 'much', 'many', 
    'well', 'way', 'even', 'also', 'back', 'go', 'going', 'want', 'did', 'do', 
    'does', 'done', 'actually', 'literally', 'thing', 'things', 'something', 
    'anything', 'nothing', 'say', 'said', 'says', 'story', 'character', 'plot'
}

def clean_text(text):
    """Cleans comment text by removing HTML, URLs, and special characters."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^\w\s]', ' ', text) # Replace punctuation with space
    text = text.lower().strip()
    return text

def get_native_sentiment(text):
    """
    Calculates sentiment score (-1 to 1) based on word matches.
    Returns 0.0 if no sentiment words are found.
    """
    words = text.split()
    pos_count = sum(1 for w in words if w in POSITIVE_WORDS)
    neg_count = sum(1 for w in words if w in NEGATIVE_WORDS)
    
    total_matches = pos_count + neg_count
    
    if total_matches == 0:
        return 0.0
    
    # Normalize score between -1 and 1
    # Formula: (Positive - Negative) / Total Matches
    return (pos_count - neg_count) / total_matches

def get_sentiment_label(score):
    """Converts score to label."""
    if score > 0.05:
        return "Positive"
    elif score < -0.05:
        return "Negative"
    else:
        return "Neutral"

# ==========================================
# 2. YouTube API Logic (Strict Filtering)
# ==========================================

def search_videos_strict(api_key, query, max_videos_to_keep=5):
    """
    Searches for videos and strictly filters out results that do not 
    contain the query string in the title.
    """
    youtube = build('youtube', 'v3', developerKey=api_key)
    
    # We fetch more results (20) to allow for filtering
    fetch_count = 20 
    
    try:
        search_response = youtube.search().list(
            q=query,
            part='id,snippet',
            maxResults=fetch_count,
            type='video',
            relevanceLanguage='en',
            order='relevance'
        ).execute()
    except Exception as e:
        st.error(f"API Error: {e}")
        return [], []

    valid_videos = []
    query_lower = query.lower().strip()

    for item in search_response.get('items', []):
        vid = item['id']['videoId']
        snippet = item['snippet']
        title = snippet['title']
        
        # --- STRICT FILTERING LOGIC ---
        if query_lower not in title.lower():
            continue
            
        video_data = {
            'video_id': vid,
            'title': title,
            'channel': snippet['channelTitle'],
            'published_at': snippet['publishedAt'],
            'thumbnail': snippet['thumbnails']['high']['url']
        }
        valid_videos.append(video_data)
        
        if len(valid_videos) >= max_videos_to_keep:
            break

    return [v['video_id'] for v in valid_videos], valid_videos

def get_video_comments(api_key, video_id, max_comments=100):
    """Fetches comments for a specific video."""
    youtube = build('youtube', 'v3', developerKey=api_key)
    comments = []
    
    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(max_comments, 100),
            textFormat="plainText",
            order="relevance"
        )
        
        while request and len(comments) < max_comments:
            response = request.execute()
            
            for item in response.get("items", []):
                comment_snippet = item["snippet"]["topLevelComment"]["snippet"]
                text = comment_snippet.get("textDisplay", "")
                published_at = comment_snippet.get("publishedAt", "")
                like_count = comment_snippet.get("likeCount", 0)
                
                comments.append({
                    "video_id": video_id,
                    "text": text,
                    "published_at": published_at,
                    "like_count": like_count
                })
                
            if len(comments) < max_comments and "nextPageToken" in response:
                request = youtube.commentThreads().list_next(request, response)
            else:
                break
                
    except Exception:
        pass
        
    return comments

def analyze_data(api_key, query, num_videos, num_comments_per_video):
    """Orchestrates the search, fetch, and analysis process."""
    
    # 1. Search
    video_ids, videos_meta = search_videos_strict(api_key, query, max_videos_to_keep=num_videos)
    
    if not video_ids:
        return None, None
    
    # 2. Fetch Comments
    all_comments = []
    progress_bar = st.progress(0)
    
    for i, vid in enumerate(video_ids):
        progress = int((i / len(video_ids)) * 100)
        progress_bar.progress(progress)
        vid_comments = get_video_comments(api_key, vid, max_comments=num_comments_per_video)
        all_comments.extend(vid_comments)
        
    progress_bar.progress(100)
    
    if not all_comments:
        return videos_meta, pd.DataFrame()

    # 3. Sentiment Analysis (Native Python)
    df = pd.DataFrame(all_comments)
    df['clean_text'] = df['text'].apply(clean_text)
    df['sentiment_score'] = df['clean_text'].apply(get_native_sentiment)
    df['sentiment_label'] = df['sentiment_score'].apply(get_sentiment_label)
    
    # 4. Date Processing
    df['published_at'] = pd.to_datetime(df['published_at'])
    if df['published_at'].dt.tz is None:
         df['published_at'] = df['published_at'].dt.tz_localize('UTC')
    df['published_at_hk'] = df['published_at'].dt.tz_convert('Asia/Hong_Kong')
    df['date'] = df['published_at_hk'].dt.date
    
    return videos_meta, df

# ==========================================
# 3. Streamlit UI
# ==========================================

st.title("🎬 Precision Movie Sentiment Analyzer")
st.markdown("""
**Problem Solved:** This tool uses a **strict title matching algorithm** to ensure that when you search for a movie, 
you don't get irrelevant news, gossip, or random trending videos.
""")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("YouTube API Key", type="password")
    movie_name = st.text_input("Movie Name", "Venom: The Last Dance")
    st.caption("Tip: Use the exact movie name for best results.")
    
    st.markdown("---")
    num_videos = st.slider("Videos to Analyze", 1, 10, 5)
    num_comments = st.slider("Comments per Video", 20, 200, 50)
    
    start_btn = st.button("Analyze", type="primary")

if start_btn and api_key and movie_name:
    with st.spinner(f"Searching strictly for '{movie_name}'..."):
        videos_meta, df_comments = analyze_data(api_key, movie_name, num_videos, num_comments)
        
        if videos_meta is None:
            st.error(f"No videos found strictly matching '{movie_name}'. Try checking your spelling.")
        elif isinstance(df_comments, pd.DataFrame) and df_comments.empty:
            st.warning("Videos found, but no comments were available.")
        else:
            # --- Display Validated Videos ---
            st.success(f"Successfully analyzed {len(videos_meta)} relevant videos.")
            
            st.subheader("✅ Verified Video Sources")
            cols = st.columns(len(videos_meta))
            for idx, vid in enumerate(videos_meta):
                with cols[idx % 3]:
                    st.image(vid['thumbnail'], use_container_width=True)
                    st.caption(f"{vid['title'][:50]}...")

            st.divider()
            
            # --- Metrics ---
            st.subheader("📊 Sentiment Overview")
            avg_sentiment = df_comments['sentiment_score'].mean()
            sentiment_counts = df_comments['sentiment_label'].value_counts()
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Comments", len(df_comments))
            c2.metric("Avg Sentiment", f"{avg_sentiment:.2f}")
            
            pos_count = sentiment_counts.get('Positive', 0)
            total_count = len(df_comments)
            ratio = (pos_count / total_count * 100) if total_count > 0 else 0
            c3.metric("Positive Ratio", f"{ratio:.1f}%")
            
            # --- Charts ---
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                # Pie Chart
                fig_pie = px.pie(
                    names=sentiment_counts.index, 
                    values=sentiment_counts.values,
                    title="Sentiment Distribution",
                    color=sentiment_counts.index,
                    color_discrete_map={"Positive": "#00CC96", "Neutral": "#636EFA", "Negative": "#EF553B"}
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                
            with col_chart2:
                # Word Cloud (Bar Chart)
                all_words = ' '.join(df_comments['clean_text']).split()
                # Filter out stopwords and short words
                filtered_words = [w for w in all_words if w not in STOPWORDS and len(w) > 2]
                word_counts = Counter(filtered_words).most_common(10)
                df_words = pd.DataFrame(word_counts, columns=['Word', 'Count'])
                
                if not df_words.empty:
                    fig_bar = px.bar(df_words, x='Count', y='Word', orientation='h', title="Top Keywords")
                    fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.info("Not enough data for keywords.")

            # --- Time Series ---
            st.subheader("📅 Sentiment Over Time")
            daily_stats = df_comments.groupby('date').agg(
                avg_sentiment=('sentiment_score', 'mean'),
                count=('sentiment_score', 'count')
            ).reset_index()
            
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(x=daily_stats['date'], y=daily_stats['avg_sentiment'], mode='lines+markers', name='Sentiment'))
            fig_line.add_trace(go.Bar(x=daily_stats['date'], y=daily_stats['count'], name='Volume', yaxis='y2', opacity=0.3))
            
            fig_line.update_layout(
                title="Daily Sentiment & Volume",
                yaxis=dict(title="Sentiment Score", range=[-1, 1]),
                yaxis2=dict(title="Volume", overlaying='y', side='right'),
                showlegend=False
            )
            st.plotly_chart(fig_line, use_container_width=True)

elif start_btn and not api_key:
    st.warning("Please enter your API Key.")
