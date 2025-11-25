import streamlit as st
import pandas as pd
from googleapiclient.discovery import build
from textblob import TextBlob
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re
import html
from datetime import datetime

# ==========================================
# 1. 設定與工具函式
# ==========================================

st.set_page_config(page_title="YouTube Movie Sentiment Analysis Pro", layout="wide")

# 停用詞列表 (可以根據需要擴充)
STOPWORDS = set([
    'the', 'a', 'an', 'in', 'on', 'at', 'of', 'to', 'is', 'are', 'was', 'were',
    'and', 'but', 'or', 'so', 'it', 'this', 'that', 'my', 'your', 'his', 'her',
    'movie', 'film', 'video', 'really', 'very', 'just', 'like', 'good', 'bad',
    'watch', 'watching', 'time', 'people', 'think', 'know', 'would', 'could',
    'should', 'get', 'got', 'make', 'made', 'see', 'saw', 'seen', 'one', 'much',
    'many', 'well', 'way', 'even', 'also', 'back', 'go', 'going', 'want',
    'did', 'do', 'does', 'done', 'actually', 'literally', 'thing', 'things',
    'something', 'anything', 'nothing', 'say', 'said', 'says', 'story', 'character',
    'characters', 'plot', 'scene', 'scenes', 'end', 'ending', 'best', 'better',
    'great', 'amazing', 'love', 'loved', 'bit', 'little', 'lot', 'movies',
    'films', 'cinema', 'actor', 'actress', 'director', 'acting', 'show', 'series'
])

def clean_text(text):
    """清理評論文字：移除 HTML 標籤、特殊符號"""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', '', text)  # 移除 HTML tag
    text = re.sub(r'http\S+', '', text)  # 移除 URL
    text = re.sub(r'[^\w\s]', '', text)  # 移除標點符號
    text = text.lower().strip()
    return text

def get_sentiment(text):
    """計算情感分數 (-1 到 1)"""
    blob = TextBlob(text)
    return blob.sentiment.polarity

def get_sentiment_label(score):
    """將分數轉換為標籤"""
    if score > 0.1:
        return "Positive"
    elif score < -0.1:
        return "Negative"
    else:
        return "Neutral"

# ==========================================
# 2. YouTube API 核心邏輯 (包含過濾機制)
# ==========================================

def search_videos_strict(api_key, query, max_results=5):
    """
    搜尋影片，並執行嚴格的標題匹配與類別過濾。
    """
    youtube = build('youtube', 'v3', developerKey=api_key)
    
    # 1. 擴大搜尋範圍：請求比 max_results 更多的影片 (例如 30 個)，以便過濾
    fetch_count = 30 
    
    search_response = youtube.search().list(
        q=query,
        part='id,snippet',
        maxResults=fetch_count,
        type='video',
        relevanceLanguage='en', # 優先搜尋英文內容 (可選)
        order='relevance'
    ).execute()

    video_ids = []
    videos_meta = []
    
    # 準備正規表達式進行不分大小寫的匹配
    # 將 query 中的空格替換為正則的 ".*" 以允許中間有其他詞 (寬鬆匹配) 或直接匹配 (嚴格匹配)
    # 這裡我們使用簡單的包含檢查
    query_lower = query.lower().strip()
    
    # 暫存初步篩選的 ID
    temp_ids = []
    temp_snippets = {}

    for item in search_response.get('items', []):
        vid = item['id']['videoId']
        title = item['snippet']['title']
        title_lower = title.lower()
        
        # --- 過濾層 1: 標題關鍵字檢查 ---
        # 檢查搜尋詞是否在標題中。
        # 如果搜尋詞是中文，直接檢查；如果是英文，檢查單詞邊界可能更準確，但這裡用簡單包含即可。
        if query_lower not in title_lower:
            # 嘗試處理 "非常盗3" vs "Now You See Me 3" 的情況
            # 如果用戶搜中文，但結果是英文，這裡會被濾掉。
            # 建議用戶輸入電影的原名或最常用的譯名。
            continue
            
        temp_ids.append(vid)
        temp_snippets[vid] = item['snippet']

    if not temp_ids:
        return [], []

    # --- 過濾層 2: 類別檢查 (Category Check) ---
    # 我們需要呼叫 videos().list 來獲取 categoryId
    videos_response = youtube.videos().list(
        id=','.join(temp_ids),
        part='snippet,statistics'
    ).execute()

    # 定義我們不想要的類別 ID (YouTube API Category IDs)
    # 25: News & Politics (新聞政治 - 這是慈濟/楊丞琳影片常出現的地方)
    # 29: Nonprofits & Activism
    # 19: Travel & Events (有時無關)
    BLOCKED_CATEGORIES = ['25', '29'] 

    filtered_videos = []

    for item in videos_response.get('items', []):
        vid = item['id']
        cat_id = item['snippet'].get('categoryId', '')
        stats = item['statistics']
        snippet = temp_snippets.get(vid, item['snippet']) # 使用 search 的 snippet 或 video 的 snippet
        
        # 排除被封鎖的類別
        if cat_id in BLOCKED_CATEGORIES:
            continue
            
        # 建立資料物件
        video_data = {
            'video_id': vid,
            'title': snippet['title'],
            'channel': snippet['channelTitle'],
            'published_at': snippet['publishedAt'], # 影片發佈時間
            'view_count': int(stats.get('viewCount', 0)),
            'like_count': int(stats.get('likeCount', 0)),
            'comment_count': int(stats.get('commentCount', 0)),
            'thumbnail': snippet['thumbnails']['high']['url']
        }
        filtered_videos.append(video_data)

    # --- 過濾層 3: 排序與截斷 ---
    # 根據觀看次數排序，取前 max_results 個
    filtered_videos.sort(key=lambda x: x['view_count'], reverse=True)
    final_videos = filtered_videos[:max_results]
    
    return [v['video_id'] for v in final_videos], final_videos

def get_video_comments(youtube, video_id, max_comments=100):
    """獲取單個影片的評論 (包含時間戳)"""
    comments = []
    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100, # 每次請求最多 100
            textFormat="plainText",
            order="relevance" 
        )
        
        while request and len(comments) < max_comments:
            response = request.execute()
            
            for item in response.get("items", []):
                comment_snippet = item["snippet"]["topLevelComment"]["snippet"]
                text = comment_snippet.get("textDisplay", "")
                published_at = comment_snippet.get("publishedAt", "") # 這是評論發佈時間
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
                
    except Exception as e:
        # 某些影片可能禁用了評論
        print(f"Error fetching comments for {video_id}: {e}")
        
    return comments

def analyze_data(api_key, query, num_videos, num_comments_per_video):
    """主分析流程"""
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    status_text.text("Step 1/4: Searching and Filtering Videos...")
    progress_bar.progress(10)
    
    # 1. 搜尋並過濾影片
    video_ids, videos_meta = search_videos_strict(api_key, query, max_results=num_videos)
    
    if not video_ids:
        status_text.text("No relevant videos found after filtering.")
        progress_bar.progress(100)
        return None, None
    
    status_text.text(f"Found {len(video_ids)} relevant videos. Step 2/4: Fetching Comments...")
    progress_bar.progress(30)
    
    # 2. 抓取評論
    youtube = build('youtube', 'v3', developerKey=api_key)
    all_comments = []
    
    for i, vid in enumerate(video_ids):
        # 更新進度
        current_progress = 30 + int((i / len(video_ids)) * 40)
        progress_bar.progress(current_progress)
        
        vid_comments = get_video_comments(youtube, vid, max_comments=num_comments_per_video)
        all_comments.extend(vid_comments)
        
    if not all_comments:
        status_text.text("No comments found on these videos.")
        progress_bar.progress(100)
        return videos_meta, pd.DataFrame()

    status_text.text("Step 3/4: Analyzing Sentiment...")
    progress_bar.progress(80)
    
    # 3. 情感分析
    df = pd.DataFrame(all_comments)
    df['clean_text'] = df['text'].apply(clean_text)
    df['sentiment_score'] = df['clean_text'].apply(get_sentiment)
    df['sentiment_label'] = df['sentiment_score'].apply(get_sentiment_label)
    
    # 4. 時間處理 (轉換為香港時間)
    df['published_at'] = pd.to_datetime(df['published_at'])
    # 轉換時區：先轉為 UTC，再轉為香港時間
    if df['published_at'].dt.tz is None:
         df['published_at'] = df['published_at'].dt.tz_localize('UTC')
    df['published_at_hk'] = df['published_at'].dt.tz_convert('Asia/Hong_Kong')
    df['date'] = df['published_at_hk'].dt.date
    
    status_text.text("Analysis Complete!")
    progress_bar.progress(100)
    status_text.empty()
    
    return videos_meta, df

# ==========================================
# 3. Streamlit UI 介面
# ==========================================

st.title("🎬 Smart Movie Review Analyzer")
st.markdown("""
This tool searches for movie reviews on YouTube, **filters out irrelevant content (like news, gossip)**, 
analyzes audience sentiment, and visualizes the trends based on **comment dates**.
""")

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter YouTube API Key", type="password")
    movie_name = st.text_input("Movie Name (e.g., Venom 3)", "Venom: The Last Dance")
    
    st.markdown("---")
    st.subheader("Advanced Settings")
    num_videos = st.slider("Number of Videos to Analyze", 1, 10, 5)
    num_comments = st.slider("Max Comments per Video", 50, 500, 100)
    
    start_btn = st.button("Start Analysis", type="primary")

if start_btn and api_key and movie_name:
    try:
        videos_meta, df_comments = analyze_data(api_key, movie_name, num_videos, num_comments)
        
        if videos_meta is None or (isinstance(df_comments, pd.DataFrame) and df_comments.empty):
            st.error(f"Could not find relevant videos or comments for '{movie_name}'. Try using the exact English title.")
        else:
            # --- 顯示影片資訊 ---
            st.subheader(f"📺 Analyzed Videos for: {movie_name}")
            st.markdown(f"These videos passed the **strict relevance filter** (Title match & Category check).")
            
            cols = st.columns(len(videos_meta))
            for idx, vid in enumerate(videos_meta):
                with cols[idx % 3]: # 簡單的排版，每行3個
                    st.image(vid['thumbnail'], use_container_width=True)
                    st.markdown(f"**{vid['title']}**")
                    st.caption(f"Channel: {vid['channel']} | Views: {vid['view_count']:,}")
            
            st.divider()
            
            # --- 1. 關鍵指標 ---
            st.subheader("📊 Sentiment Overview")
            avg_sentiment = df_comments['sentiment_score'].mean()
            sentiment_counts = df_comments['sentiment_label'].value_counts()
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Comments", len(df_comments))
            col2.metric("Average Sentiment", f"{avg_sentiment:.2f}", 
                        delta="Positive" if avg_sentiment > 0 else "Negative")
            col3.metric("Positive Comments", sentiment_counts.get("Positive", 0))
            col4.metric("Negative Comments", sentiment_counts.get("Negative", 0))
            
            # --- 2. 情感分佈圓餅圖 ---
            fig_pie = px.pie(
                names=sentiment_counts.index, 
                values=sentiment_counts.values,
                title="Sentiment Distribution",
                color=sentiment_counts.index,
                color_discrete_map={"Positive": "#00CC96", "Neutral": "#636EFA", "Negative": "#EF553B"}
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # --- 3. 每日情感趨勢 (基於評論日期) ---
            st.subheader("📅 Daily Sentiment Trend (Based on Comment Date)")
            st.info("This chart shows when people commented, not when the video was uploaded.")
            
            # 聚合數據：按日期計算平均情感分數和評論數量
            daily_stats = df_comments.groupby('date').agg(
                avg_sentiment=('sentiment_score', 'mean'),
                comment_count=('sentiment_score', 'count')
            ).reset_index()
            
            # 建立雙軸圖表
            fig_trend = go.Figure()
            
            # 長條圖：評論數量
            fig_trend.add_trace(go.Bar(
                x=daily_stats['date'],
                y=daily_stats['comment_count'],
                name='Comment Volume',
                marker_color='rgba(200, 200, 200, 0.5)',
                yaxis='y2'
            ))
            
            # 線圖：情感分數
            fig_trend.add_trace(go.Scatter(
                x=daily_stats['date'],
                y=daily_stats['avg_sentiment'],
                name='Avg Sentiment',
                mode='lines+markers',
                line=dict(color='#636EFA', width=3)
            ))
            
            fig_trend.update_layout(
                title="Sentiment & Volume Over Time",
                xaxis_title="Date (Hong Kong Time)",
                yaxis=dict(title="Sentiment Score (-1 to 1)", range=[-1, 1]),
                yaxis2=dict(title="Number of Comments", overlaying='y', side='right', showgrid=False),
                legend=dict(x=0, y=1.1, orientation='h'),
                hovermode="x unified"
            )
            st.plotly_chart(fig_trend, use_container_width=True)
            
            # --- 4. 文字雲 (使用頻率統計模擬) ---
            st.subheader("☁️ Most Frequent Words")
            
            # 簡單的詞頻統計
            all_words = ' '.join(df_comments['clean_text']).split()
            filtered_words = [w for w in all_words if w not in STOPWORDS and len(w) > 2]
            word_counts = Counter(filtered_words).most_common(20)
            
            df_words = pd.DataFrame(word_counts, columns=['Word', 'Count'])
            
            fig_bar = px.bar(
                df_words, 
                x='Count', 
                y='Word', 
                orientation='h',
                title="Top 20 Words in Comments",
                color='Count',
                color_continuous_scale='Viridis'
            )
            fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # --- 5. 數據表格 ---
            with st.expander("View Raw Data"):
                st.dataframe(df_comments[['date', 'text', 'sentiment_label', 'sentiment_score', 'like_count']])

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please check your API Key and internet connection.")

elif start_btn and not api_key:
    st.warning("Please enter your YouTube API Key.")
