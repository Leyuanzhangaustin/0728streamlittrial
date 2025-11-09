# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime, timedelta
import time
import asyncio
import openai
import re
from opencc import OpenCC
from googleapiclient.discovery import build

# =========================
# 0. 工具函數：關鍵字、語言檢測
# =========================

def generate_search_queries(movie_title: str):
    """
    生成更寬鬆且多樣化的關鍵字組合：
    - 去掉過於嚴格的引號限制
    - 中英混合，涵蓋「影評/評論/評價/解析/分析/無雷/有雷/預告/花絮/反應/review/reaction/ending explained」
    - 仍保留少量帶引號的精確匹配，作為補充
    """
    zh_terms = [
        "影評", "評論", "評價", "點評", "解析", "分析", "觀後感",
        "無雷", "有雷", "討論", "好唔好睇", "預告", "花絮", "片段", "首映", "幕後"
    ]
    en_terms = [
        "review", "reaction", "ending explained", "analysis", "explained",
        "behind the scenes", "bts", "premiere", "interview", "press conference"
    ]

    # 寬鬆（無引號）
    loose = [f"{movie_title}"]
    loose += [f"{movie_title} {t}" for t in zh_terms]
    loose += [f"{movie_title} {t}" for t in en_terms]

    # 少量精確（帶引號）
    tight = [
        f"\"{movie_title}\"",
        f"\"{movie_title}\" 影評",
        f"\"{movie_title}\" 評論",
        f"\"{movie_title}\" 解析",
        f"\"{movie_title}\" review",
        f"\"{movie_title}\" reaction",
    ]

    # 去重保序
    seen = set()
    queries = []
    for q in loose + tight:
        if q not in seen:
            queries.append(q)
            seen.add(q)
    return queries


def count_chars(text: str):
    """
    計算各類字符數量：中日韓漢字、假名、拉丁、數字等
    """
    counts = {
        "cjk": 0,
        "hiragana": 0,
        "katakana": 0,
        "half_katakana": 0,
        "hangul": 0,
        "latin": 0,
        "digits": 0,
        "other": 0
    }
    for ch in text:
        code = ord(ch)
        if 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF or 0xF900 <= code <= 0xFAFF:
            counts["cjk"] += 1
        elif 0x3040 <= code <= 0x309F:
            counts["hiragana"] += 1
        elif 0x30A0 <= code <= 0x30FF or 0x31F0 <= code <= 0x31FF:
            counts["katakana"] += 1
        elif 0xFF65 <= code <= 0xFF9F:
            counts["half_katakana"] += 1
        elif 0xAC00 <= code <= 0xD7AF:
            counts["hangul"] += 1
        elif (0x0041 <= code <= 0x005A) or (0x0061 <= code <= 0x007A):
            counts["latin"] += 1
        elif 0x0030 <= code <= 0x0039:
            counts["digits"] += 1
        else:
            counts["other"] += 1
    return counts


def diff_chars(a: str, b: str) -> int:
    """
    估算字符級差異數：zip 對齊後不等 + 長度差
    """
    m = min(len(a), len(b))
    base = sum(1 for i in range(m) if a[i] != b[i])
    return base + abs(len(a) - len(b))


def classify_zh_trad_simp(text: str, cc_t2s: OpenCC, cc_s2t: OpenCC):
    """
    簡單的語言/書寫系統分類：
    - ja：含有較高比例的假名（平/片/半角片假名）
    - zh-Hant：t2s 變化顯著而 s2t 變化很小（原文更接近繁體）
    - zh-Hans：相反（原文更接近簡體）
    - zh-unkn：中文但難以區分（多為公共漢字+標點）
    - other：基本沒有 CJK
    """
    if not isinstance(text, str) or len(text.strip()) < 2:
        return "other"

    counts = count_chars(text)
    kana = counts["hiragana"] + counts["katakana"] + counts["half_katakana"]
    cjk = counts["cjk"]

    # 日文剔除：假名數 >= 2 且相對占比 > 10% 視為日文
    if kana >= 2 and kana / max(1, (cjk + kana)) >= 0.10:
        return "ja"

    if cjk < 1:
        return "other"

    t2s = cc_t2s.convert(text)  # 繁->簡
    s2t = cc_s2t.convert(text)  # 簡->繁
    ct2s = diff_chars(text, t2s)
    cs2t = diff_chars(text, s2t)

    threshold = max(1, int(0.05 * cjk))  # cjk 的 5% 作為差異閾值

    if ct2s > cs2t + threshold:
        return "zh-Hant"
    elif cs2t > ct2s + threshold:
        return "zh-Hans"
    else:
        return "zh-unkn"


# =========================
# 1. YouTube 搜尋（強化 + 分頁）
# =========================

def search_youtube_videos(
    keywords,
    youtube_client,
    max_per_keyword,
    start_date,
    end_date,
    add_language_bias=True
):
    """
    - 對每個關鍵字用 order=relevance 與 order=viewCount 兩種排序抓取
    - 分頁直到達到 max_per_keyword 或無更多結果
    - 返回：
      - video_ids: 去重後的所有視頻 ID 列表
      - video_meta: {video_id: {"title": ..., "channelTitle": ..., "publishedAt": ...}}
    """
    all_video_ids = set()
    video_meta = {}

    for query in keywords:
        collected_for_query = set()
        for order in ["relevance", "viewCount"]:
            try:
                request = youtube_client.search().list(
                    q=query,
                    part="id,snippet",
                    type="video",
                    maxResults=50,  # API 上限
                    publishedAfter=f"{start_date}T00:00:00Z",
                    publishedBefore=f"{end_date}T23:59:59Z",
                    order=order,
                    safeSearch="none",
                    **({"relevanceLanguage": "zh-Hant"} if add_language_bias else {})
                )
                while request and len(collected_for_query) < max_per_keyword:
                    response = request.execute()
                    for item in response.get("items", []):
                        vid = item["id"]["videoId"]
                        if vid in collected_for_query:
                            continue
                        collected_for_query.add(vid)
                        all_video_ids.add(vid)
                        # 保留一份基本元數據（標題/頻道/時間）
                        if vid not in video_meta:
                            snip = item.get("snippet", {})
                            video_meta[vid] = {
                                "title": snip.get("title", ""),
                                "channelTitle": snip.get("channelTitle", ""),
                                "publishedAt": snip.get("publishedAt", "")
                            }
                    # 翻頁
                    request = youtube_client.search().list_next(request, response)
                    if len(collected_for_query) >= max_per_keyword:
                        break
                    time.sleep(0.2)  # 溫和限流
            except Exception as e:
                st.warning(f"搜尋關鍵字 '{query}'（order={order}）時發生錯誤: {e}")
                continue
    return list(all_video_ids), video_meta


# =========================
# 2. 批量抓取留言（補充來源信息）
# =========================

def get_all_comments(video_ids, youtube_client, max_per_video, video_meta=None):
    """
    抓取每個視頻的頂層評論（commentThreads），直到達到每視頻上限。
    為每條評論添加來源視頻的標題與超鏈接。
    """
    video_meta = video_meta or {}
    all_comments = []
    total_videos = len(video_ids)
    progress_bar = st.progress(0, text="抓取 YouTube 留言中...")

    for i, video_id in enumerate(video_ids):
        try:
            request = youtube_client.commentThreads().list(
                part="snippet",
                videoId=video_id,
                textFormat="plainText",
                order="time",
                maxResults=100
            )
            comments_fetched = 0
            while request and comments_fetched < max_per_video:
                response = request.execute()
                for item in response.get("items", []):
                    if comments_fetched >= max_per_video:
                        break
                    comment = item["snippet"]["topLevelComment"]["snippet"]
                    all_comments.append({
                        "video_id": video_id,
                        "video_title": video_meta.get(video_id, {}).get("title", ""),
                        "video_url": f"https://www.youtube.com/watch?v={video_id}",
                        "comment_text": comment.get("textDisplay", ""),
                        "published_at": comment.get("publishedAt", ""),
                        "like_count": comment.get("likeCount", 0)
                    })
                    comments_fetched += 1
                if comments_fetched >= max_per_video:
                    break
                request = youtube_client.commentThreads().list_next(request, response)
                time.sleep(0.2)
        except Exception:
            # 有些視頻可能關閉了評論或被限權
            pass
        finally:
            progress_bar.progress(
                (i + 1) / max(1, total_videos),
                text=f"抓取 YouTube 留言中... ({i+1}/{total_videos} 部影片)"
            )
    progress_bar.empty()
    return pd.DataFrame(all_comments)


# =========================
# 3. DeepSeek AI 異步情感分析（順序對齊）
# =========================

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
                data = response.choices[0].message.content
                analysis_result = json.loads(data)
                return analysis_result
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return {"sentiment": "Error", "topic": "Error", "summary": f"API Error: {e}"}


async def run_all_analyses(df, deepseek_client):
    """
    as_completed + 進度條，但用索引回填，確保輸出與輸入順序一一對齊
    """
    semaphore = asyncio.Semaphore(50)
    tasks = []

    async def with_index(idx, text):
        res = await analyze_comment_deepseek_async(text, deepseek_client, semaphore)
        return idx, res

    for i, text in enumerate(df["comment_text"]):
        tasks.append(asyncio.create_task(with_index(i, text)))

    progress_bar = st.progress(0, text="AI 情感分析中...")
    results = [None] * len(tasks)
    for done_idx, coro in enumerate(asyncio.as_completed(tasks), start=1):
        idx, res = await coro
        results[idx] = res
        progress_bar.progress(done_idx / len(tasks), text=f"AI 情感分析中... ({done_idx}/{len(tasks)})")
    progress_bar.empty()
    return results


# =========================
# 4. 主流程（增強：語言過濾、來源字段、搜尋加寬）
# =========================

def movie_comment_analysis(
    movie_title, start_date, end_date,
    yt_api_key, deepseek_api_key,
    max_videos_per_keyword=30, max_comments_per_video=50, sample_size=None,
    relax_trad_filter=True
):
    # 關鍵字生成：寬鬆+精確混合
    SEARCH_KEYWORDS = generate_search_queries(movie_title)

    youtube_client = build("youtube", "v3", developerKey=yt_api_key)
    deepseek_client = openai.AsyncOpenAI(
        api_key=deepseek_api_key,
        base_url="https://api.deepseek.com/v1"
    )

    # 搜尋（分頁 + 元數據）
    video_ids, video_meta = search_youtube_videos(
        SEARCH_KEYWORDS, youtube_client, max_videos_per_keyword, start_date, end_date, add_language_bias=True
    )
    if not video_ids:
        return None, "找不到相關影片。"

    # 留言抓取（補視頻標題與鏈接）
    df_comments = get_all_comments(video_ids, youtube_client, max_comments_per_video, video_meta=video_meta)
    if df_comments.empty:
        return None, "找不到任何留言。"

    # 語言過濾：顯式剔除日文，保留繁體（可選保留「疑似中文」）
    st.info(f"已抓取 {len(df_comments)} 則原始留言，現開始語言篩選（繁體 + 剔除日文）...")

    cc_t2s = OpenCC("t2s")  # 繁->簡
    cc_s2t = OpenCC("s2t")  # 簡->繁

    def lang_pred(text):
        return classify_zh_trad_simp(text, cc_t2s, cc_s2t)

    df_comments["lang_pred"] = df_comments["comment_text"].apply(lang_pred)

    if relax_trad_filter:
        # 放寬：保留 zh-Hant + zh-unkn（疑似中文但難判繁/簡），剔除 ja/other/zh-Hans
        df_comments_filtered = df_comments[df_comments["lang_pred"].isin(["zh-Hant", "zh-unkn"])].reset_index(drop=True)
    else:
        # 嚴格：只保留 zh-Hant
        df_comments_filtered = df_comments[df_comments["lang_pred"] == "zh-Hant"].reset_index(drop=True)

    # 顯式剔除日文
    df_comments_filtered = df_comments_filtered[df_comments_filtered["lang_pred"] != "ja"]

    st.info(f"篩選後剩下 {len(df_comments_filtered)} 則符合條件的留言。")
    if df_comments_filtered.empty:
        return None, "在抓取的留言中找不到符合語言條件的內容。"

    # 時區處理與日期範圍二次校驗
    df_comments_filtered["published_at"] = pd.to_datetime(df_comments_filtered["published_at"], utc=True, errors="coerce")
    df_comments_filtered["published_at_hk"] = df_comments_filtered["published_at"].dt.tz_convert("Asia/Hong_Kong")

    start_dt = pd.to_datetime(start_date).tz_localize("Asia/Hong_Kong")
    end_dt = pd.to_datetime(end_date).tz_localize("Asia/Hong_Kong") + timedelta(days=1)
    mask_date = (df_comments_filtered["published_at_hk"] >= start_dt) & (df_comments_filtered["published_at_hk"] < end_dt)
    df_comments_filtered = df_comments_filtered.loc[mask_date].reset_index(drop=True)
    if df_comments_filtered.empty:
        return None, "在指定日期範圍內沒有符合語言條件的留言。"

    # 取樣控制
    if sample_size and 0 < sample_size < len(df_comments_filtered):
        df_analyze = df_comments_filtered.sample(n=sample_size, random_state=42)
    else:
        df_analyze = df_comments_filtered

    st.info(f"準備對 {len(df_analyze)} 則留言進行高速並發分析...")

    # 異步分析並對齊
    analysis_results = asyncio.run(run_all_analyses(df_analyze, deepseek_client))
    analysis_df = pd.DataFrame(analysis_results)

    final_df = pd.concat([df_analyze.reset_index(drop=True), analysis_df], axis=1)
    final_df["published_at"] = pd.to_datetime(final_df["published_at"])

    return final_df, None


# =========================
# 5. Streamlit UI
# =========================

st.set_page_config(page_title="YouTube 電影評論 AI 分析", layout="wide")
st.title("🎬 YouTube 電影評論 AI 情感分析")

with st.expander("使用說明"):
    st.markdown("""
    1.  輸入電影的中文全名、分析時間範圍及所需的 API 金鑰。
    2.  自訂每個關鍵字搜尋的影片數量上限，及每部影片抓取的留言數量上限。
    3.  系統將自動抓取 YouTube 留言，剔除日文，並以繁體為主要目標語言進行 AI 情感分析。
        你可選擇是否「放寬繁體判定」，以增加樣本量。
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
max_videos = st.slider("每個關鍵字的最大影片搜尋數", 5, 80, 30, help="增加此數值會找到更多影片，但會增加 YouTube API 的配額消耗。")
max_comments = st.slider("每部影片的最大留言抓取數", 10, 200, 80, help="數量越多，分析結果越全面，但 DeepSeek API 成本越高。")
sample_size = st.number_input("分析留言數量上限 (0 代表分析全部已抓取的留言)", 0, 5000, 500, help="例如抓取了 2000 則留言，這裡設 500 就只會分析其中的 500 則。")
relax_trad_filter = st.checkbox("放寬繁體判定（允許疑似中文但無法判別繁／簡的留言）", value=True)

if st.button("🚀 開始分析"):
    if not all([movie_title, yt_api_key, deepseek_api_key]):
        st.warning("請填寫電影名稱和兩個 API 金鑰。")
    else:
        result_container = st.container()
        with st.spinner("AI 高速分析中... 請稍候..."):
            df_result, err = movie_comment_analysis(
                movie_title, str(start_date), str(end_date),
                yt_api_key, deepseek_api_key,
                max_videos, max_comments, sample_size,
                relax_trad_filter=relax_trad_filter
            )

        if err:
            st.error(err)
        else:
            st.success("分析完成！")
            st.dataframe(df_result.head(20), use_container_width=True)

            st.header("📊 可視化分析結果")

            # 共用設定
            sentiments_order = ['Positive', 'Negative', 'Neutral', 'Invalid', 'Error']
            colors_map = {
                'Positive': '#5cb85c', 'Negative': '#d9534f', 'Neutral': '#f0ad4e',
                'Invalid': '#cccccc', 'Error': '#888888'
            }

            # 1. 情感分佈圓餅圖
            st.subheader("1. Sentiment Distribution (Pie)")
            sentiment_series = df_result['sentiment'].dropna().astype(str)
            sentiment_counts = sentiment_series.value_counts()
            ordered_labels = [label for label in sentiments_order if label in sentiment_counts.index]

            if not sentiment_counts.empty:
                fig1 = px.pie(
                    values=sentiment_counts[ordered_labels].values,
                    names=ordered_labels,
                    title='Overall Sentiment Distribution',
                    color=ordered_labels,
                    color_discrete_map=colors_map,
                    hole=0.0
                )
                st.plotly_chart(fig1, use_container_width=True)
            else:
                st.info("No sentiment data available for pie chart.")

            # 2. 每日情感趨勢圖
            st.subheader("2. Daily Sentiment Trend")

            if 'published_at_hk' in df_result.columns:
                df_result['date'] = df_result['published_at_hk'].dt.date
            else:
                df_result['date'] = pd.to_datetime(df_result['published_at'], utc=True).dt.tz_convert('Asia/Hong_Kong').dt.date

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

            # 3. 各主題情感佔比
            st.subheader("3. Sentiment Share by Topic")
            topic_sentiment = df_result.groupby(['topic', 'sentiment']).size().unstack().fillna(0)
            topic_sentiment = topic_sentiment.reindex(columns=sentiments_order).dropna(axis=1, how='all')

            if not topic_sentiment.empty:
                topic_sentiment = topic_sentiment[topic_sentiment.sum(axis=1) > 0]
                if not topic_sentiment.empty:
                    topic_sentiment_percent = topic_sentiment.div(topic_sentiment.sum(axis=1), axis=0).fillna(0) * 100
                    fig3 = px.bar(
                        topic_sentiment_percent.reset_index().melt(id_vars='topic', var_name='sentiment', value_name='pct'),
                        x='topic', y='pct', color='sentiment',
                        title='Sentiment Share by Topic',
                        labels={'topic': 'Topic', 'pct': 'Percentage (%)', 'sentiment': 'Sentiment'},
                        color_discrete_map=colors_map
                    )
                    st.plotly_chart(fig3, use_container_width=True)
                else:
                    st.info("No topic data with comments to display the chart.")
            else:
                st.info("Not enough topic sentiment data to display the stacked bar chart.")

            # 4. 下載分析明細（新增 video_title / video_url）
            st.subheader("4. 下載分析明細")
            csv = df_result.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                "📥 下載全部分析明細 (CSV)",
                csv,
                file_name=f"{movie_title}_analysis_details.csv",
                mime='text/csv'
            )
