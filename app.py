# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import re

# ========== 检查是否为繁体中文 ==========
def is_traditional_chinese(text):
    """
    Returns True if text is mostly Traditional Chinese.
    Simple heuristic: if there are more Traditional than Simplified chars, treat as Traditional.
    """
    # 常见简体字集合
    simplified_chars = set("们体为产举乐乡书买乱争亚伪众优会传伤伦体价众优伤伦体价众优伤伦体价众优伤伦体价众优伤伦体价众优伤伦体价众优伤伦体价众优伤伦体价众优伤伦体价众优伤伦体价众优伤伦体价众优伤伦体价众优伤伦体价众优伤伦体价众优伤伦体价")
    traditional_chars = set("們體為產舉樂鄉書買亂爭亞偽眾優會傳傷倫體價")
    # 统计出现的繁简体字符数
    simplified_count = sum(1 for c in text if c in simplified_chars)
    traditional_count = sum(1 for c in text if c in traditional_chars)
    # 若繁体字符多于简体字符，则视为繁体
    return traditional_count >= simplified_count and traditional_count > 0

# ========== 1. YouTube 搜索 ==========
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

# ========== 2. 批量抓取评论 ==========
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

# ========== 3. DeepSeek AI情感分析 ==========
def analyze_comment_deepseek(comment_text, deepseek_client, max_retries=3):
    import json
    if not isinstance(comment_text, str) or len(comment_text.strip()) < 5:
        return {"sentiment": "Invalid", "topic": "N/A", "summary": "Comment too short or invalid."}
    system_prompt = (
        "你是一位专业的香港市场舆情分析师。请分析以下电影评论，并严格按照JSON格式返回结果。"
        "JSON对象必须包含三个键："
        "1. 'sentiment': 其值必须是 'Positive', 'Negative', 或 'Neutral'。"
        "2. 'topic': 评论讨论的核心主题，例如 '剧情', '演员演技', '动作设计', '画面美术', '电影节奏', '整体感觉'。如果无法判断，则为 'N/A'。"
        "3. 'summary': 用一句话简潔总结评论的核心观点。"
        "确保输出只有JSON对象，无任何额外文字。"
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

# ========== 4. 主流程 ==========
def movie_comment_analysis(
    movie_title, start_date, end_date,
    yt_api_key, deepseek_api_key,
    max_videos_per_keyword=30, max_comments_per_video=50, sample_size=None
):
    # 关键词
    SEARCH_KEYWORDS = [
        f'"{movie_title}" 預告',
        f'"{movie_title}" 影評',
        f'"{movie_title}" 分析',
        f'"{movie_title}" 好唔好睇',
        f'"{movie_title}" 討論',
        f'"{movie_title}" reaction'
    ]
    # API初始化
    from googleapiclient.discovery import build
    youtube_client = build('youtube', 'v3', developerKey=yt_api_key)
    import openai
    deepseek_client = openai.OpenAI(
        api_key=deepseek_api_key,
        base_url="https://api.deepseek.com/v1"
    )

    # 搜索视频
    video_ids = search_youtube_videos(
        SEARCH_KEYWORDS, youtube_client, max_videos_per_keyword, start_date, end_date
    )
    if not video_ids:
        return None, "未找到相关视频"
    # 抓取评论
    df_comments = get_all_comments(video_ids, youtube_client, max_comments_per_video)
    if df_comments.empty:
        return None, "没有抓到评论"
    # 时间处理
    df_comments['published_at'] = pd.to_datetime(df_comments['published_at'], utc=True)
    df_comments['published_at_hk'] = df_comments['published_at'].dt.tz_convert('Asia/Hong_Kong')
    # 按香港时区时间过滤
    start = pd.to_datetime(start_date).tz_localize('Asia/Hong_Kong')
    end = pd.to_datetime(end_date).tz_localize('Asia/Hong_Kong') + timedelta(days=1)
    mask = (df_comments['published_at_hk'] >= start) & (df_comments['published_at_hk'] <= end)
    df_comments = df_comments.loc[mask].reset_index(drop=True)
    if df_comments.empty:
        return None, "没有符合日期范围的评论"
    # ------ 新增: 过滤非繁体评论 ------
    df_comments = df_comments[df_comments['comment_text'].apply(is_traditional_chinese)].reset_index(drop=True)
    if df_comments.empty:
        return None, "没有符合条件的繁体中文评论"
    # 抽样
    if sample_size and sample_size < len(df_comments):
        df_analyze = df_comments.sample(n=sample_size, random_state=42)
    else:
        df_analyze = df_comments

    # AI情感分析
    from tqdm import tqdm
    tqdm.pandas(desc="AI情感分析")
    analysis_results = df_analyze['comment_text'].progress_apply(
        lambda x: analyze_comment_deepseek(x, deepseek_client)
    )
    analysis_df = pd.json_normalize(analysis_results)
    final_df = pd.concat([df_analyze.reset_index(drop=True), analysis_df], axis=1)
    final_df['published_at'] = pd.to_datetime(final_df['published_at'])
    return final_df, None

# ========== 5. Streamlit UI ==========
st.set_page_config(page_title="电影YouTube评论AI分析", layout="wide")
st.title("🎬 YouTube 电影评论AI情感分析")

with st.expander("操作说明"):
    st.markdown("""
    1. 输入电影名称、分析时间范围、API KEY。
    2. 可自定义每组关键词最大视频数及每个视频最大评论数。
    3. 点击“开始分析”按钮，自动抓取评论，逐条调用AI分析情感与主题。
    4. 分析完成后可浏览可视化结果，并下载明细。
    """)

movie_title = st.text_input("电影名称", value="")
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("起始日期", value=datetime.today() - timedelta(days=30))
with col2:
    end_date = st.date_input("结束日期", value=datetime.today())
yt_api_key = st.text_input("YouTube API Key", type='password')
deepseek_api_key = st.text_input("DeepSeek API Key", type='password')
max_videos = st.slider("每组关键词最多视频数", 5, 50, 10)
max_comments = st.slider("每个视频最多评论数", 5, 100, 20)
sample_size = st.number_input("最多分析评论数（0为全量）", 0, 2000, 0)

if st.button("🚀 开始分析"):
    if not all([movie_title, yt_api_key, deepseek_api_key]):
        st.warning("请填写所有内容。")
    else:
        with st.spinner("AI分析中，请耐心等待...（如评论数多，需数分钟）"):
            df_result, err = movie_comment_analysis(
                movie_title, str(start_date), str(end_date),
                yt_api_key, deepseek_api_key,
                max_videos, max_comments, sample_size or None
            )
        if err:
            st.error(err)
        else:
            st.success("分析完成！")
            st.dataframe(df_result.head(20))

            # ========== 可视化 (Visualization in English) ==========
            st.subheader("1. Sentiment Distribution")
            fig1, ax1 = plt.subplots(figsize=(4, 4))
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

else:
    st.info("请填写信息并点击“开始分析”")
