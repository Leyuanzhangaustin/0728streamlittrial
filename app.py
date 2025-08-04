# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import json
import openai
from googleapiclient.discovery import build
pip install opencc-python-reimplemented
# ==============================================================================
# 1. 優化後的關鍵詞生成策略
# ==============================================================================
def get_optimized_keywords(movie_title, director_name=None, actor_names=None):
    """
    動態生成一組更全面的搜索關鍵詞，以覆蓋更多香港電影相關討論。
    """
    # 基礎關鍵詞 (核心部分)
    base_keywords = [
        f'"{movie_title}" 預告', f'"{movie_title}" 影評', f'"{movie_title}" 觀後感',
        f'"{movie_title}" 分析', f'"{movie_title}" 解析', f'"{movie_title}" 解讀', f'"{movie_title}" 心得',
    ]
    # 口語化及網絡俚語 (捕捉地道反應)
    colloquial_keywords = [
        f'"{movie_title}" 好唔好睇', f'"{movie_title}" 伏唔伏', f'"{movie_title}" 有冇伏',
        f'"{movie_title}" 吹水', f'"{movie_title}" 討論', f'"{movie_title}" reaction', f'"{movie_title}" 吐槽',
    ]
    # 內容形式 (覆蓋不同角度的影片)
    format_keywords = [
        f'"{movie_title}" 懶人包', f'"{movie_title}" 彩蛋', f'"{movie_title}" 幕後花絮',
        f'"{movie_title}" 製作特輯', f'"{movie_title}" 訪問',
    ]
    # 上映週期與事件 (捕捉特定時間點的熱度)
    event_keywords = [
        f'"{movie_title}" 上映', f'"{movie_title}" 首映', f'"{movie_title}" 優先場', f'"{movie_title}" 謝票場',
    ]
    # 負面及爭議性關鍵詞 (確保數據平衡)
    negative_keywords = [
        f'"{movie_title}" 負評', f'"{movie_title}" 劣評', f'"{movie_title}" 中伏',
        f'"{movie_title}" 失望', f'"{movie_title}" 爛片', f'"{movie_title}" 爭議',
    ]
    # 純標題搜索
    title_only_keywords = [f'"{movie_title}"']
    # 動態關鍵詞 (選填，但強烈建議)
    dynamic_keywords = []
    if director_name:
        dynamic_keywords.append(f'"{director_name}" "{movie_title}"')
    if actor_names:
        for actor in actor_names:
            dynamic_keywords.append(f'"{actor}" "{movie_title}"')

    all_keywords = (
        base_keywords + colloquial_keywords + format_keywords + 
        event_keywords + negative_keywords + title_only_keywords + dynamic_keywords
    )
    return list(set(all_keywords))

# ==============================================================================
# 2. YouTube 搜索 (已移除地理位置限制)
# ==============================================================================
def search_youtube_videos(keywords, youtube_client, max_per_keyword, start_date, end_date):
    all_video_ids = set()
    for query in keywords:
        nextPageToken = None
        fetched = 0
        while fetched < max_per_keyword:
            try:
                remaining = max_per_keyword - fetched
                max_fetch = min(50, remaining)
                search_response = youtube_client.search().list(
                    q=query,
                    part='id,snippet',
                    type='video',
                    maxResults=max_fetch,
                    publishedAfter=f"{start_date}T00:00:00Z",
                    publishedBefore=f"{end_date}T23:59:59Z",
                    relevanceLanguage='zh-Hant', # 優先返回繁體中文內容
                    # regionCode='HK', # 已根據您的要求移除地理位置限制
                    pageToken=nextPageToken
                ).execute()
                video_ids = [item['id']['videoId'] for item in search_response.get('items', [])]
                all_video_ids.update(video_ids)
                fetched += len(video_ids)
                nextPageToken = search_response.get('nextPageToken')
                if not nextPageToken or len(video_ids) == 0:
                    break
                time.sleep(0.5)
            except Exception as e:
                st.warning(f"搜索關鍵詞 '{query}' 時發生錯誤: {e}")
                break
    return list(all_video_ids)

# ==============================================================================
# 3. 批量抓取评论
# ==============================================================================
def get_all_comments(video_ids, youtube_client, max_per_video):
    all_comments = []
    progress_bar = st.progress(0)
    for i, video_id in enumerate(video_ids):
        try:
            request = youtube_client.commentThreads().list(
                part='snippet', videoId=video_id, textFormat='plainText', maxResults=100
            )
            comments_fetched = 0
            while request and comments_fetched < max_per_video:
                response = request.execute()
                for item in response['items']:
                    if comments_fetched >= max_per_video: break
                    comment = item['snippet']['topLevelComment']['snippet']
                    all_comments.append({
                        'video_id': video_id,
                        'comment_text': comment['textDisplay'],
                        'published_at': comment['publishedAt'],
                        'like_count': comment['likeCount']
                    })
                    comments_fetched += 1
                if comments_fetched >= max_per_video: break
                request = youtube_client.commentThreads().list_next(request, response)
        except Exception as e:
            st.warning(f"抓取影片 {video_id} 的評論時跳過，原因: {e}")
            continue
        finally:
            progress_bar.progress((i + 1) / len(video_ids), text=f"正在抓取影片評論 ({i+1}/{len(video_ids)})")
    progress_bar.empty()
    return pd.DataFrame(all_comments)

# ==============================================================================
# 4. DeepSeek AI情感分析
# ==============================================================================
def analyze_comment_deepseek(comment_text, deepseek_client, max_retries=3):
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

# ==============================================================================
# 5. 主流程 (已加入繁體中文過濾)
# ==============================================================================
def movie_comment_analysis(
    search_keywords, start_date, end_date,
    yt_api_key, deepseek_api_key,
    max_videos_per_keyword, max_comments_per_video, sample_size
):
    # API初始化
    youtube_client = build('youtube', 'v3', developerKey=yt_api_key)
    deepseek_client = openai.OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com/v1")

    # 搜索视频
    st.info(f"正在使用 {len(search_keywords)} 組優化關鍵詞搜索相關影片...")
    video_ids = search_youtube_videos(search_keywords, youtube_client, max_videos_per_keyword, start_date, end_date)
    if not video_ids:
        return None, "未找到相關視頻，請嘗試放寬日期或調整電影名稱。"
    st.success(f"找到 {len(video_ids)} 個相關影片，開始抓取評論...")

    # 抓取评论
    df_comments = get_all_comments(video_ids, youtube_client, max_comments_per_video)
    if df_comments.empty:
        return None, "成功找到影片，但未能抓取到任何評論。"
    
    # === 新增：繁體中文評論過濾 ===
    try:
        from opencc import OpenCC
        cc = OpenCC('t2s.json')
        def is_traditional_chinese(text):
            if not isinstance(text, str) or not text.strip(): return False
            return cc.convert(text) != text
        
        original_count = len(df_comments)
        df_comments = df_comments[df_comments['comment_text'].apply(is_traditional_chinese)].copy()
        st.info(f"從 {original_count} 條原始評論中，篩選出 {len(df_comments)} 條繁體中文評論。")
        if df_comments.empty:
            return None, "未檢測到繁體中文評論。"
    except ImportError:
        st.error("缺少 'opencc-python-reimplemented' 庫，無法進行繁體中文過濾。請運行 `pip install opencc-python-reimplemented`。")
        return None, "缺少必要組件。"
    # ==============================

    # 时间处理
    df_comments['published_at'] = pd.to_datetime(df_comments['published_at'], utc=True)
    df_comments['published_at_hk'] = df_comments['published_at'].dt.tz_convert('Asia/Hong_Kong')
    start_dt = pd.to_datetime(start_date).tz_localize('Asia/Hong_Kong')
    end_dt = pd.to_datetime(end_date).tz_localize('Asia/Hong_Kong') + timedelta(days=1)
    mask = (df_comments['published_at_hk'] >= start_dt) & (df_comments['published_at_hk'] <= end_dt)
    df_comments = df_comments.loc[mask].reset_index(drop=True)
    if df_comments.empty:
        return None, "在指定日期範圍內沒有找到繁體中文評論。"

    # 抽样
    if sample_size and sample_size < len(df_comments):
        df_analyze = df_comments.sample(n=sample_size, random_state=42)
        st.info(f"從 {len(df_comments)} 條符合條件的評論中，隨機抽樣 {sample_size} 條進行分析。")
    else:
        df_analyze = df_comments
        st.info(f"將對全部 {len(df_comments)} 條符合條件的評論進行分析。")

    # AI情感分析
    from tqdm import tqdm
    tqdm.pandas(desc="AI情感分析")
    analysis_results = df_analyze['comment_text'].progress_apply(lambda x: analyze_comment_deepseek(x, deepseek_client))
    analysis_df = pd.json_normalize(analysis_results)
    final_df = pd.concat([df_analyze.reset_index(drop=True), analysis_df], axis=1)
    final_df['published_at'] = pd.to_datetime(final_df['published_at'])
    return final_df, None

# ==============================================================================
# 6. Streamlit UI (已更新參數和輸入項)
# ==============================================================================
st.set_page_config(page_title="电影YouTube评论AI分析", layout="wide")
st.title("🎬 YouTube 电影评论AI情感分析（香港市场版）")

with st.expander("ℹ️ 操作說明與建議"):
    st.markdown("""
    1.  **輸入電影的香港官方譯名**，以獲得最精準的搜索結果。
    2.  **（可選）輸入導演和主要演員姓名**，能覆蓋更多圍繞人物展開的討論影片。
    3.  **調整參數**：想獲得更多評論，請**調高“每組關鍵詞最多視頻數”**和**“每個視頻最多評論數”**。
    4.  **API Keys**：請填入您自己的 YouTube Data API v3 和 DeepSeek API 的密鑰。
    5.  **開始分析**：點擊按鈕後，程式會自動完成：`搜索影片` -> `抓取評論` -> `過濾繁體字` -> `AI逐條分析` -> `生成圖表`。過程可能需要數分鐘，請耐心等候。
    """)

# --- 輸入欄位 ---
st.header("1. 輸入分析目標")
movie_title = st.text_input("電影名稱（香港譯名）", value="九龍城寨之圍城")
director_name = st.text_input("導演名稱（可選）", value="鄭保瑞")
actors_str = st.text_input("主要演員（用逗號/空格分隔，可選）", value="古天樂, 林峯, 劉俊謙")

st.header("2. 設定分析範圍與密鑰")
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("起始日期", value=datetime.today() - timedelta(days=30))
with col2:
    end_date = st.date_input("結束日期", value=datetime.today())
yt_api_key = st.text_input("YouTube API Key", type='password')
deepseek_api_key = st.text_input("DeepSeek API Key", type='password')

st.header("3. 調整數據抓取量")
max_videos = st.slider("每組關鍵詞最多視頻數（越高，來源越廣）", 5, 100, 30)
max_comments = st.slider("每個視頻最多評論數（越高，評論越多）", 10, 500, 50)
sample_size = st.number_input("最多分析評論數（設為 0 則分析全部）", 0, 5000, 0)

# --- 分析按鈕與主邏輯 ---
if st.button("🚀 開始分析", use_container_width=True):
    if not all([movie_title, yt_api_key, deepseek_api_key]):
        st.warning("請務必填寫電影名稱和兩個 API Key。")
    else:
        # 動態生成關鍵詞
        actor_names = [name.strip() for name in actors_str.replace('，', ',').replace(' ', ',').split(',') if name.strip()]
        SEARCH_KEYWORDS = get_optimized_keywords(movie_title, director_name, actor_names)
        
        with st.spinner("AI分析中，請耐心等待...（如評論數多，需數分鐘）"):
            df_result, err = movie_comment_analysis(
                SEARCH_KEYWORDS, str(start_date), str(end_date),
                yt_api_key, deepseek_api_key,
                max_videos, max_comments, sample_size or None
            )
        
        if err:
            st.error(f"分析中斷：{err}")
        else:
            st.success("分析完成！")
            st.header("📊 分析結果概覽")

            # --- 數據展示 ---
            st.subheader("部分原始數據預覽")
            st.dataframe(df_result.head(20))

            # --- 可視化 ---
            st.subheader("1. 整體情感分佈")
            fig1, ax1 = plt.subplots(figsize=(5, 4))
            sentiment_counts = df_result['sentiment'].value_counts()
            colors_map = {"Positive": "#5cb85c", "Negative": "#d9534f", "Neutral": "#f0ad4e", "Invalid": "#cccccc", "Error": "#888888"}
            pie_colors = [colors_map.get(s, "#333333") for s in sentiment_counts.index]
            sentiment_counts.plot.pie(autopct='%.1f%%', ax=ax1, colors=pie_colors, startangle=90)
            ax1.set_title('Sentiment Distribution')
            ax1.set_ylabel('')
            st.pyplot(fig1)

            st.subheader("2. 每日情感趨勢")
            df_result['date'] = df_result['published_at_hk'].dt.date
            daily = df_result.groupby(['date', 'sentiment']).size().unstack().fillna(0)
            # 確保欄位順序一致，方便上色
            daily = daily.reindex(columns=['Positive', 'Negative', 'Neutral', 'Invalid', 'Error'], fill_value=0)
            
            col_chart, line_chart = st.columns(2)
            with col_chart:
                fig2, ax2 = plt.subplots(figsize=(8, 4))
                daily[['Positive', 'Negative', 'Neutral']].plot(
                    kind='bar', stacked=True, ax=ax2, width=0.8, 
                    color=[colors_map['Positive'], colors_map['Negative'], colors_map['Neutral']]
                )
                ax2.set_title('每日情感趨勢 (堆疊長條圖)')
                ax2.set_xlabel('日期')
                ax2.set_ylabel('評論數量')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig2)

            with line_chart:
                fig3, ax3 = plt.subplots(figsize=(8, 4))
                for sentiment in ['Positive', 'Negative', 'Neutral']:
                    if sentiment in daily.columns:
                        ax3.plot(daily.index, daily[sentiment], marker='o', linestyle='-', label=sentiment, color=colors_map[sentiment])
                ax3.set_title("每日情感趨勢 (折線圖)")
                ax3.set_xlabel("日期")
                ax3.set_ylabel("評論數量")
                ax3.legend(title="Sentiment")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig3)

            st.subheader("3. 核心主題討論佔比")
            fig4, ax4 = plt.subplots(figsize=(8, 5))
            # 過濾掉無效主題
            topic_counts = df_result[~df_result['topic'].isin(['N/A', 'Error'])]['topic'].value_counts()
            topic_counts.plot(kind='barh', ax=ax4, color='#0288d1')
            ax4.set_title('核心討論主題 Top 10')
            ax4.set_xlabel('評論數量')
            # 在長條圖上顯示數字
            for index, value in enumerate(topic_counts):
                ax4.text(value, index, f' {value}')
            plt.tight_layout()
            st.pyplot(fig4)

            # --- 下載按鈕 ---
            st.header("4. 下載分析明細")
            csv = df_result.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 下載全部分析明細 (CSV)",
                data=csv,
                file_name=f"{movie_title}_youtube_analysis_{start_date}_to_{end_date}.csv",
                mime='text/csv',
                use_container_width=True
            )
else:
    st.info("請填寫頂部信息並點擊“開始分析”以生成報告。")
