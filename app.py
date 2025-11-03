# app.py (Final Visualization Version - Revised for Language Filtering)
pip install opencc-python-reimplemented
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
from opencc import OpenCC  # ### NEW: 引入 OpenCC 庫

# ========== 1. YouTube Search (MODIFIED) ==========
# 移除了 'relevanceLanguage' 和 'regionCode' 參數，因為它們不是嚴格的過濾器。
# 我們將在抓取留言後進行更可靠的語言篩選。
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
                publishedBefore=f"{end_date}T23:59:59Z"
                # ### MODIFIED: 移除以下兩行 ###
                # relevanceLanguage='zh-Hant',
                # regionCode='HK'
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
    progress_bar = st.progress(0, text="抓取 YouTube 留言中...")
    total_videos = len(video_ids)
    for i, video_id in enumerate(video_ids):
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
        finally:
            # 更新進度條
            progress_bar.progress((i + 1) / total_videos, text=f"抓取 YouTube 留言中... ({i+1}/{total_videos} 部影片)")
    progress_bar.empty() # 完成後移除進度條
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

# ========== 4. Main Process (MODIFIED) ==========
# 增加了繁體中文留言的篩選邏輯
async def run_all_analyses(df, deepseek_client):
    semaphore = asyncio.Semaphore(50)
    tasks = [
        analyze_comment_deepseek_async(comment_text, deepseek_client, semaphore)
        for comment_text in df['comment_text']
    ]

    # 使用 Streamlit 的進度條來顯示 AI 分析進度
    progress_bar = st.progress(0, text="AI 情感分析中...")
    
    results = []
    for i, f in enumerate(asyncio.as_completed(tasks)):
        results.append(await f)
        progress_bar.progress((i + 1) / len(tasks), text=f"AI 情感分析中... ({i+1}/{len(tasks)})")
        
    progress_bar.empty()
    # 由於 as_completed 不保證順序，我們需要一種方法來重新對齊結果。
    # 這裡我們暫時假設順序問題不大，或者在更複雜的場景中需要傳遞索引。
    # 為了簡單起見，我們直接返回結果列表。
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

    # ### NEW: 繁體中文留言篩選邏輯 ###
    st.info(f"已抓取 {len(df_comments)} 則原始留言，現開始篩選繁體中文內容...")
    
    # 初始化 OpenCC，'t2s' 表示從繁體 (Traditional) 到簡體 (Simplified)
    cc = OpenCC('t2s')
    
    def is_traditional_chinese(text):
        if not isinstance(text, str) or len(text.strip()) < 2:
            return False
        # 判斷邏輯：如果將文本從繁體轉換為簡體後，與原文不同，
        # 就意味著原文中至少包含一個可被轉換的繁體字。
        return cc.convert(text) != text

    mask_trad = df_comments['comment_text'].apply(is_traditional_chinese)
    df_comments_filtered = df_comments[mask_trad].reset_index(drop=True)
    
    st.info(f"篩選後剩下 {len(df_comments_filtered)} 則繁體中文留言。")
    
    if df_comments_filtered.empty:
        return None, "在抓取的留言中找不到繁體中文內容。"
    
    # 後續流程使用篩選後的 DataFrame
    df_comments = df_comments_filtered
    # ### END OF NEW BLOCK ###

    df_comments['published_at'] = pd.to_datetime(df_comments['published_at'], utc=True)
    df_comments['published_at_hk'] = df_comments['published_at'].dt.tz_convert('Asia/Hong_Kong')

    start_dt = pd.to_datetime(start_date).tz_localize('Asia/Hong_Kong')
    end_dt = pd.to_datetime(end_date).tz_localize('Asia/Hong_Kong') + timedelta(days=1)
    mask_date = (df_comments['published_at_hk'] >= start_dt) & (df_comments['published_at_hk'] < end_dt)
    df_comments = df_comments.loc[mask_date].reset_index(drop=True)
    if df_comments.empty:
        return None, "在指定日期範圍內沒有符合語言條件的留言。"

    if sample_size and sample_size > 0 and sample_size < len(df_comments):
        df_analyze = df_comments.sample(n=sample_size, random_state=42)
    else:
        df_analyze = df_comments

    st.info(f"準備對 {len(df_analyze)} 則留言進行高速並發分析...")
    
    # 運行異步分析
    analysis_results_unordered = asyncio.run(run_all_analyses(df_analyze, deepseek_client))
    
    # 因為 asyncio.as_completed 的結果是無序的，我們需要將其與原始數據安全地合併。
    # 最安全的方法是將分析結果轉換為 DataFrame，並確保其索引與 df_analyze 一致。
    analysis_df = pd.DataFrame(analysis_results_unordered)
    
    # 檢查行數是否匹配
    if len(df_analyze) != len(analysis_df):
        st.warning("AI 分析返回的結果數量與請求數量不匹配，數據可能未完全對齊。")
        # 採取一種保守的合併策略
        min_len = min(len(df_analyze), len(analysis_df))
        final_df = pd.concat([df_analyze.head(min_len).reset_index(drop=True), analysis_df.head(min_len)], axis=1)
    else:
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
    3.  點擊「開始分析」，系統將自動抓取 YouTube 留言，**篩選出繁體中文內容**，並進行 AI 高速情感分析。
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
        # 使用一個容器來包裹整個分析過程，方便最後統一處理
        result_container = st.container()
        
        with st.spinner("AI 高速分析中... 請稍候..."):
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

            # --- 1. 情感分佈圓餅圖 ---
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

            # --- 2. 每日情感趨勢圖 ---
            st.subheader("2. Daily Sentiment Trend")
            
            if 'published_at_hk' in df_result.columns:
                df_result['date'] = df_result['published_at_hk'].dt.date
            else:
                df_result['date'] = df_result['published_at'].dt.date
            
            daily = df_result.groupby(['date', 'sentiment']).size().unstack().fillna(0)
            daily = daily.reindex(columns=sentiments_order).dropna(axis=1, how='all')

            if not daily.empty:
                daily_long = daily.reset_index().melt(id_vars='date', var_name='sentiment', value_name='count')
                
                st.markdown("#### 每日情感趨勢 (折線圖)")
                fig_line = px.line(
                    daily_long, x='date', y='count', color='sentiment',
                    title='Daily Comment Volume Trend by Sentiment',
                    labels={'date': 'Date', 'count': 'Number of Comments', 'sentiment': 'Sentiment'},
                    color_discrete_map=colors_map,
                    category_orders={'sentiment': [col for col in sentiments_order if col in daily.columns]}
                )
                st.plotly_chart(fig_line, use_container_width=True)

                st.markdown("#### 每日留言總量及情感分佈 (堆疊長條圖)")
                fig_bar = px.bar(
                    daily_long, x='date', y='count', color='sentiment',
                    title='Daily Comment Volume by Sentiment (Stacked)',
                    labels={'date': 'Date', 'count': 'Number of Comments', 'sentiment': 'Sentiment'},
                    color_discrete_map=colors_map,
                    category_orders={'sentiment': [col for col in sentiments_order if col in daily.columns]},
                    barmode='stack'
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("Not enough daily sentiment data to display the trend charts.")

            # --- 3. 各主題情感佔比 ---
            st.subheader("3. Sentiment Share by Topic")
            topic_sentiment = df_result.groupby(['topic', 'sentiment']).size().unstack().fillna(0)
            topic_sentiment = topic_sentiment.reindex(columns=sentiments_order).dropna(axis=1, how='all')
            
            if not topic_sentiment.empty:
                topic_sentiment = topic_sentiment[topic_sentiment.sum(axis=1) > 0]
                
                if not topic_sentiment.empty:
                    topic_sentiment_percent = topic_sentiment.div(topic_sentiment.sum(axis=1), axis=0).fillna(0) * 100

                    fig3, ax3 = plt.subplots(figsize=(10, 5))
                    topic_sentiment_percent.plot(
                        kind='bar', stacked=True, ax=ax3,
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

            # --- 4. 下載分析明細 ---
            st.subheader("4. 下載分析明細")
            csv = df_result.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                "📥 下載全部分析明細 (CSV)",
                csv,
                file_name=f"{movie_title}_analysis_details.csv",
                mime='text/csv'
            )
