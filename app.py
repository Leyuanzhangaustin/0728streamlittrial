import asyncio
import json
import math
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple, Any, Optional

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from opencc import OpenCC
from openai import AsyncOpenAI


# =========================
# 0. 常數與工具函數
# =========================

CANTONESE_CHAR_TOKENS = {
    "呢", "嗰", "咁", "喺", "嘅", "咗", "喇", "啦", "唔", "冇", "嚟", "乜",
    "佢", "佬", "啱", "噉", "嘢", "瞓", "攞", "講", "齋", "梗", "冧", "揾",
    "陣", "齡", "餸", "曬", "啲", "啩", "嗱", "啱", "冇", "噃", "喺", "嚟",
    "喺", "勁", "咁", "哋", "係", "嗰", "吓", "喱", "嚟", "嗰", "嗌", "滾"
}

CANTONESE_PHRASES = [
    "唔該", "點算", "好正", "好鍾意", "好勁", "有冇", "冇問題", "好掂",
    "傾偈", "行街", "出嚟", "飲茶", "煩唔煩", "玩吓", "揾食", "識唔識",
    "真係", "得唔得", "咁上下", "咁點", "講真", "笑死", "好睇", "引喎",
    "頂唔順", "搞掂", "唔錯喎", "扮嘢", "亂噏", "痴線", "冇所謂", "正呀",
    "靚到", "覺得好", "都幾", "忍唔住", "快啲", "慢慢嚟", "俾面", "收皮"
]

HK_KEYWORD_PATTERNS = [
    "香港", "Hong Kong", "HK", "港片", "港產", "粵語", "粤语", "Cantonese", "廣東話",
    "港人", "港味", "港式", "九龍", "旺角", "銅鑼灣", "中環", "尖沙咀", "太古", "沙田"
]

NEGATIVE_CHANNEL_HINTS = ["china", "cn", "mainland", "官方", "央視"]


def is_cjk(char: str) -> bool:
    """Return True if character is a CJK Unified Ideograph."""
    return "\u4e00" <= char <= "\u9fff"


def count_zh_chars(text: str) -> int:
    """Count Chinese characters in the given text."""
    return sum(1 for ch in text if is_cjk(ch))


def diff_chars(original: str, converted: str) -> int:
    """Count the number of differing characters between two strings of equal length."""
    length = min(len(original), len(converted))
    return sum(1 for i in range(length) if original[i] != converted[i])


def classify_zh_trad_simp(text: str, cc_t2s: OpenCC, cc_s2t: OpenCC) -> str:
    """
    Classify text into Traditional Chinese, Simplified Chinese, or unknown/other.
    Returns: "zh-Hant", "zh-Hans", "zh-unkn", "other", "ja".
    """
    stripped = (text or "").strip()
    if not stripped:
        return "other"

    zh_char_count = count_zh_chars(stripped)
    if zh_char_count == 0:
        return "other"

    trad = cc_s2t.convert(stripped)
    simp = cc_t2s.convert(stripped)

    trad_diff = diff_chars(stripped, trad)
    simp_diff = diff_chars(stripped, simp)

    # Heuristic thresholds
    threshold = max(1, math.ceil(zh_char_count * 0.05))

    if trad_diff <= threshold < simp_diff:
        return "zh-Hant"
    if simp_diff <= threshold < trad_diff:
        return "zh-Hans"

    # Check for Japanese (presence of Hiragana/Katakana)
    if any("\u3040" <= ch <= "\u30ff" for ch in stripped):
        return "ja"

    return "zh-unkn"


def score_cantonese(text: str) -> float:
    """
    Assign a heuristic Cantonese score between 0 and 10 based on unique characters and phrases.
    """
    if not text:
        return 0.0

    zh_chars = [ch for ch in text if is_cjk(ch)]
    total_zh = len(zh_chars)
    if total_zh == 0:
        return 0.0

    unique_hits = sum(1 for ch in zh_chars if ch in CANTONESE_CHAR_TOKENS)
    phrase_hits = sum(text.count(phrase) for phrase in CANTONESE_PHRASES)

    char_score = unique_hits / total_zh
    phrase_score = min(phrase_hits * 0.6, 4.0)
    combined = min(char_score * 6 + phrase_score, 10.0)

    return round(combined, 3)


def compute_hk_video_score(
    video_id: str,
    video_meta: Dict[str, Dict[str, Any]],
    video_extra: Dict[str, Dict[str, Any]],
    channel_country_map: Dict[str, Optional[str]]
) -> float:
    """
    Estimate how likely a video is relevant to Hong Kong audiences.
    """
    meta = video_meta.get(video_id, {})
    extra = video_extra.get(video_id, {})
    title = meta.get("title", "") or ""
    channel_title = meta.get("channelTitle", "") or ""
    tags = extra.get("tags", []) or []
    channel_id = extra.get("channelId")
    channel_country = channel_country_map.get(channel_id)

    score = 0.0
    combined_text = " ".join([title] + tags).lower()

    for pattern in HK_KEYWORD_PATTERNS:
        if pattern.lower() in combined_text:
            score += 0.8

    if channel_country == "HK":
        score += 2.5
    elif channel_country in {"CN", "CN-TW", "TW"}:
        score += 0.4
    elif channel_country:
        score += 0.2

    lowered_channel_title = channel_title.lower()
    if any(hint in lowered_channel_title for hint in NEGATIVE_CHANNEL_HINTS):
        score -= 0.5

    if "粵語" in combined_text or "cantonese" in combined_text:
        score += 1.0

    # Boost if title contains full-width punctuation typical in Cantonese media
    if any(char in title for char in {"！", "～", "「", "」"}):
        score += 0.2

    return round(score, 3)


def generate_search_queries(movie_title: str) -> List[str]:
    """
    Optimized keyword generation using Boolean OR operators to reduce API quota usage.
    """
    base = movie_title.strip()
    if not base:
        return []

    keywords_group = "影評|評論|香港|粵語|review|reaction|HK|trailer"
    q1 = base
    q2 = f"\"{base}\" ({keywords_group})"
    return [q1, q2]


# =========================
# 1. YouTube API 搜尋與快取
# =========================

@st.cache_data(ttl=3600, show_spinner=False)
def search_youtube_videos_cached(
    _youtube_client,
    keywords: List[str],
    max_per_keyword: int,
    start_date: str,
    end_date: str,
    add_language_bias: bool = True,
    region_bias: bool = True
) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    """
    Optimized YouTube search function with caching to minimize quota usage.
    """
    all_video_ids: set = set()
    video_meta: Dict[str, Dict[str, Any]] = {}

    for query in keywords:
        if not query:
            continue

        collected_for_query: set = set()

        try:
            request = _youtube_client.search().list(
                q=query,
                part="id,snippet",
                type="video",
                maxResults=50,
                publishedAfter=f"{start_date}T00:00:00Z",
                publishedBefore=f"{end_date}T23:59:59Z",
                order="relevance",
                safeSearch="none",
                **({"relevanceLanguage": "zh-Hant"} if add_language_bias else {}),
                **({"regionCode": "HK"} if region_bias else {})
            )

            current_page = 0
            pages_limit = 2

            while request and len(collected_for_query) < max_per_keyword and current_page < pages_limit:
                response = request.execute()
                for item in response.get("items", []):
                    vid = item.get("id", {}).get("videoId")
                    if not vid or vid in collected_for_query:
                        continue

                    collected_for_query.add(vid)
                    all_video_ids.add(vid)

                    snippet = item.get("snippet", {}) or {}
                    video_meta.setdefault(vid, {
                        "title": snippet.get("title", ""),
                        "channelTitle": snippet.get("channelTitle", ""),
                        "publishedAt": snippet.get("publishedAt", ""),
                    })

                if len(collected_for_query) >= max_per_keyword:
                    break

                request = _youtube_client.search().list_next(request, response)
                current_page += 1

        except HttpError as http_err:
            print(f"[YouTube API] HTTP error for query '{query}': {http_err}")
        except Exception as exc:
            print(f"[YouTube API] Unexpected error for query '{query}': {exc}")

    return list(all_video_ids), video_meta


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_video_and_channel_details_cached(
    _youtube_client,
    video_ids: List[str]
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Optional[str]]]:
    """
    Fetch video details (language metadata, tags, etc.) and channel country info.
    """
    video_extra: Dict[str, Dict[str, Any]] = {}
    channel_ids: set = set()

    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i:i + 50]
        try:
            resp = _youtube_client.videos().list(
                part="snippet,contentDetails",
                id=",".join(chunk)
            ).execute()

            for item in resp.get("items", []):
                vid = item.get("id")
                snippet = item.get("snippet", {}) or {}
                ch_id = snippet.get("channelId")
                default_language = snippet.get("defaultLanguage") or ""
                default_audio_language = snippet.get("defaultAudioLanguage") or ""
                tags = snippet.get("tags", []) or []

                video_extra[vid] = {
                    "channelId": ch_id,
                    "defaultLanguage": default_language,
                    "defaultAudioLanguage": default_audio_language,
                    "tags": tags
                }

                if ch_id:
                    channel_ids.add(ch_id)

        except HttpError as http_err:
            print(f"[YouTube API] HTTP error when fetching video details: {http_err}")
        except Exception as exc:
            print(f"[YouTube API] Unexpected error when fetching video details: {exc}")

    channel_country: Dict[str, Optional[str]] = {}

    ids = list(channel_ids)
    for i in range(0, len(ids), 50):
        chunk = ids[i:i + 50]
        try:
            resp = _youtube_client.channels().list(
                part="brandingSettings",
                id=",".join(chunk)
            ).execute()

            for item in resp.get("items", []):
                cid = item.get("id")
                branding = (item.get("brandingSettings", {}) or {}).get("channel", {}) or {}
                country = branding.get("country")
                channel_country[cid] = country

        except HttpError as http_err:
            print(f"[YouTube API] HTTP error when fetching channel details: {http_err}")
        except Exception as exc:
            print(f"[YouTube API] Unexpected error when fetching channel details: {exc}")

    return video_extra, channel_country


@st.cache_data(ttl=3600, show_spinner=False)
def get_all_comments_cached(
    _youtube_client,
    video_ids: List[str],
    max_per_video: int,
    video_meta: Dict[str, Dict[str, Any]],
    hk_score_map: Dict[str, float],
    video_extra: Dict[str, Dict[str, Any]],
    channel_country_map: Dict[str, Optional[str]]
) -> pd.DataFrame:
    """
    Retrieve comments for each video with caching to avoid repeated API calls.
    """
    records: List[Dict[str, Any]] = []

    for video_id in video_ids:
        try:
            request = _youtube_client.commentThreads().list(
                part="snippet",
                videoId=video_id,
                textFormat="plainText",
                order="time",
                maxResults=100
            )

            fetched = 0
            while request and fetched < max_per_video:
                response = request.execute()
                for item in response.get("items", []):
                    if fetched >= max_per_video:
                        break

                    snippet = item.get("snippet", {}) or {}
                    top_comment = snippet.get("topLevelComment", {}) or {}
                    comment_snippet = top_comment.get("snippet", {}) or {}

                    video_info = video_meta.get(video_id, {})
                    extra_info = video_extra.get(video_id, {})
                    channel_id = extra_info.get("channelId")

                    records.append({
                        "video_id": video_id,
                        "video_title": video_info.get("title", ""),
                        "video_channel_title": video_info.get("channelTitle", ""),
                        "video_url": f"https://www.youtube.com/watch?v={video_id}",
                        "video_published_at": video_info.get("publishedAt", ""),
                        "video_hk_score": hk_score_map.get(video_id, 0.0),
                        "video_channel_id": channel_id,
                        "video_channel_country": channel_country_map.get(channel_id) if channel_id else None,
                        "video_default_audio_lang": extra_info.get("defaultAudioLanguage", ""),
                        "comment_text": comment_snippet.get("textDisplay", ""),
                        "comment_published_at": comment_snippet.get("publishedAt", ""),
                        "comment_like_count": comment_snippet.get("likeCount", 0)
                    })
                    fetched += 1

                if fetched >= max_per_video:
                    break

                request = _youtube_client.commentThreads().list_next(request, response)

        except HttpError as http_err:
            print(f"[YouTube API] HTTP error fetching comments for video {video_id}: {http_err}")
        except Exception as exc:
            print(f"[YouTube API] Unexpected error fetching comments for video {video_id}: {exc}")

    return pd.DataFrame(records)


# =========================
# 2. DeepSeek 異步評論分析
# =========================

async def analyze_comment_deepseek_async(
    client: Optional[AsyncOpenAI],
    comment_text: str,
    video_title: str,
    video_url: str
) -> Dict[str, Any]:
    """
    Analyze a comment asynchronously using DeepSeek (through OpenAI-compatible API).
    """
    default_result = {
        "analysis_sentiment": None,
        "analysis_sentiment_confidence": None,
        "analysis_tags": [],
        "analysis_summary": None
    }

    if client is None or not comment_text.strip():
        return default_result

    system_prompt = (
        "You are an analytical assistant specializing in Cantonese YouTube comments. "
        "Respond in valid JSON. Use Traditional Chinese for summaries and tags. "
        "Recognize colloquial Cantonese expressions."
    )

    user_prompt = (
        "請分析以下粵語或華語 YouTube 留言，輸出 JSON 物件，格式如下：\n"
        "{\n"
        '  "sentiment": "positive|neutral|negative",\n'
        '  "confidence": 0-1 (float),\n'
        '  "tags": ["<關鍵字1>", "<關鍵字2>", ...],\n'
        '  "summary": "以繁體中文撰寫的一句話摘要"\n'
        "}\n\n"
        f"影片標題: {video_title}\n"
        f"影片連結: {video_url}\n"
        f"留言內容: {comment_text}\n"
    )

    try:
        response = await client.chat.completions.create(
            model="deepseek-chat",
            temperature=0.2,
            max_tokens=400,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        content = (response.choices[0].message.content or "").strip()

        parsed = json.loads(content)
        sentiment = parsed.get("sentiment")
        confidence = parsed.get("confidence")
        tags = parsed.get("tags", [])
        summary = parsed.get("summary")

        if isinstance(tags, str):
            tags = [tags]

        default_result.update({
            "analysis_sentiment": sentiment,
            "analysis_sentiment_confidence": confidence,
            "analysis_tags": tags,
            "analysis_summary": summary
        })

    except json.JSONDecodeError:
        default_result["analysis_summary"] = content or None
    except Exception as exc:
        print(f"[DeepSeek] Error analyzing comment: {exc}")

    return default_result


async def run_all_analyses(
    df: pd.DataFrame,
    client: Optional[AsyncOpenAI],
    concurrency: int = 5
) -> List[Dict[str, Any]]:
    """
    Run DeepSeek analyses concurrently with bounded concurrency.
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_analyze(row) -> Dict[str, Any]:
        async with semaphore:
            return await analyze_comment_deepseek_async(
                client,
                row.comment_text,
                row.video_title,
                row.video_url
            )

    tasks = [bounded_analyze(row) for row in df.itertuples()]
    return await asyncio.gather(*tasks)


def run_async(coro):
    """
    Safely execute asyncio coroutines in environments where an event loop may already exist.
    """
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            asyncio.set_event_loop(None)
            loop.close()


# =========================
# 3. 主分析流程
# =========================

def movie_comment_analysis(
    movie_title: str,
    start_date: str,
    end_date: str,
    yt_api_key: str,
    deepseek_api_key: Optional[str],
    max_videos_per_keyword: int = 30,
    max_comments_per_video: int = 50,
    sample_size: Optional[int] = None,
    relax_trad_filter: bool = True,
    cantonese_threshold: float = 2.0,
    auto_relax_threshold: bool = True,
    target_min_cantonese: int = 300,
    prefer_hk_videos: bool = True
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Execute the full pipeline: search videos, fetch metadata, gather comments,
    filter by language/Cantonese score, and run DeepSeek analyses.
    """
    keywords = generate_search_queries(movie_title)
    if not keywords:
        return None, "請提供有效的影片或電影名稱。"

    youtube_client = build("youtube", "v3", developerKey=yt_api_key)
    deepseek_client = None

    if deepseek_api_key:
        deepseek_client = AsyncOpenAI(
            api_key=deepseek_api_key,
            base_url="https://api.deepseek.com/v1"
        )

    video_ids, video_meta = search_youtube_videos_cached(
        youtube_client,
        keywords,
        max_videos_per_keyword,
        start_date,
        end_date,
        add_language_bias=True,
        region_bias=True
    )

    if not video_ids:
        return None, "找不到相關影片，請嘗試調整關鍵字或日期範圍。"

    video_extra, channel_country_map = fetch_video_and_channel_details_cached(
        youtube_client,
        video_ids
    )

    hk_score_map = {
        vid: compute_hk_video_score(vid, video_meta, video_extra, channel_country_map)
        for vid in video_ids
    }

    if prefer_hk_videos:
        video_ids_sorted = sorted(
            video_ids,
            key=lambda v: hk_score_map.get(v, 0.0),
            reverse=True
        )
    else:
        video_ids_sorted = video_ids

    df_comments = get_all_comments_cached(
        youtube_client,
        video_ids_sorted,
        max_comments_per_video,
        video_meta,
        hk_score_map,
        video_extra,
        channel_country_map
    )

    if df_comments.empty:
        return None, "找不到任何影片留言，請放寬條件或延長日期範圍。"

    df_comments["lang_pred"] = None
    df_comments["cantonese_score"] = 0.0

    cc_t2s = OpenCC("t2s")
    cc_s2t = OpenCC("s2t")

    df_comments = df_comments.copy()
    df_comments["lang_pred"] = df_comments["comment_text"].apply(
        lambda text: classify_zh_trad_simp(text, cc_t2s, cc_s2t)
    )

    if relax_trad_filter:
        df_comments = df_comments[
            df_comments["lang_pred"].isin(["zh-Hant", "zh-unkn"])
        ].reset_index(drop=True)
    else:
        df_comments = df_comments[
            df_comments["lang_pred"] == "zh-Hant"
        ].reset_index(drop=True)

    if df_comments.empty:
        return None, "語言篩選後無資料，請調整條件。"

    df_comments["cantonese_score"] = df_comments["comment_text"].apply(score_cantonese)

    threshold = float(cantonese_threshold)
    df_filtered = df_comments[df_comments["cantonese_score"] >= threshold].reset_index(drop=True)

    if auto_relax_threshold and len(df_filtered) < target_min_cantonese:
        new_threshold = threshold
        while len(df_filtered) < target_min_cantonese and new_threshold > 0.5:
            new_threshold = round(new_threshold - 0.5, 2)
            df_filtered = df_comments[df_comments["cantonese_score"] >= new_threshold].reset_index(drop=True)
        threshold = new_threshold

    if df_filtered.empty:
        return None, "粵語篩選後樣本為零，請調整門檻或延長日期範圍。"

    df_filtered["comment_published_at"] = pd.to_datetime(
        df_filtered["comment_published_at"],
        utc=True,
        errors="coerce"
    )
    df_filtered["comment_published_at_hk"] = df_filtered["comment_published_at"].dt.tz_convert("Asia/Hong_Kong")

    start_dt = pd.to_datetime(start_date).tz_localize("Asia/Hong_Kong")
    end_dt = pd.to_datetime(end_date).tz_localize("Asia/Hong_Kong") + timedelta(days=1)

    mask = (df_filtered["comment_published_at_hk"] >= start_dt) & (
        df_filtered["comment_published_at_hk"] < end_dt
    )
    df_filtered = df_filtered.loc[mask].reset_index(drop=True)

    if df_filtered.empty:
        return None, "在指定日期範圍內沒有符合粵語條件的留言。"

    if sample_size and 0 < sample_size < len(df_filtered):
        df_analyze = df_filtered.sample(n=sample_size, random_state=42).reset_index(drop=True)
    else:
        df_analyze = df_filtered.reset_index(drop=True)

    analysis_results = run_async(run_all_analyses(df_analyze, deepseek_client)) if deepseek_client else [
        {
            "analysis_sentiment": None,
            "analysis_sentiment_confidence": None,
            "analysis_tags": [],
            "analysis_summary": None
        }
        for _ in range(len(df_analyze))
    ]

    analysis_df = pd.DataFrame(analysis_results)
    final_df = pd.concat([df_analyze, analysis_df], axis=1)

    final_df["comment_published_at"] = pd.to_datetime(
        final_df["comment_published_at"], errors="coerce"
    )
    final_df["comment_published_at_hk"] = pd.to_datetime(
        final_df["comment_published_at_hk"], errors="coerce"
    )

    final_df["cantonese_threshold_applied"] = threshold
    final_df["movie_title_query"] = movie_title
    final_df["analysis_timestamp"] = pd.Timestamp.utcnow().tz_convert("Asia/Hong_Kong")

    final_df.sort_values(
        by=["video_hk_score", "cantonese_score", "comment_like_count"],
        ascending=[False, False, False],
        inplace=True
    )

    return final_df.reset_index(drop=True), None


# =========================
# 4. 視覺化
# =========================

def render_visualizations(df: pd.DataFrame) -> None:
    st.markdown("### 📈 視覺化洞察")

    if df.empty:
        st.info("沒有資料可視覺化。")
        return

    # 情緒分佈
    sentiment_counts = (
        df["analysis_sentiment"]
        .fillna("未分析")
        .value_counts()
        .reset_index()
        .rename(columns={"index": "sentiment", "analysis_sentiment": "count"})
    )

    sentiment_chart = (
        alt.Chart(sentiment_counts)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("sentiment:N", title="情緒分類"),
            y=alt.Y("count:Q", title="留言數"),
            color=alt.Color(
                "sentiment:N",
                scale=alt.Scale(
                    domain=["positive", "neutral", "negative", "未分析"],
                    range=["#16a34a", "#facc15", "#dc2626", "#9ca3af"]
                ),
                legend=None
            ),
            tooltip=["sentiment:N", "count:Q"]
        )
    ).properties(
        width="container",
        height=320,
        title="留言情緒分佈"
    )

    st.altair_chart(sentiment_chart, use_container_width=True)

    # 日期趨勢
    date_df = df.dropna(subset=["comment_published_at_hk"]).copy()
    if not date_df.empty:
        date_df["comment_date"] = date_df["comment_published_at_hk"].dt.date
        daily_counts = (
            date_df.groupby("comment_date")
            .size()
            .reset_index(name="count")
        )

        daily_chart = (
            alt.Chart(daily_counts)
            .mark_area(line={"color": "#2563eb"}, color="#2563eb40")
            .encode(
                x=alt.X("comment_date:T", title="日期"),
                y=alt.Y("count:Q", title="每日留言數"),
                tooltip=[alt.Tooltip("comment_date:T", title="日期"), alt.Tooltip("count:Q", title="留言數")]
            )
        ).properties(
            width="container",
            height=320,
            title="每日留言量趨勢（香港時間）"
        )

        st.altair_chart(daily_chart, use_container_width=True)
    else:
        st.info("留言缺少時間資訊，無法顯示日期趨勢。")

    # 熱門標籤
    tag_series = (
        df["analysis_tags"]
        .dropna()
        .explode()
        .astype(str)
        .str.strip()
    )
    tag_series = tag_series[tag_series != ""]
    if not tag_series.empty:
        top_tags = (
            tag_series.value_counts()
            .head(15)
            .reset_index()
            .rename(columns={"index": "tag", "analysis_tags": "count"})
        )

        tag_chart = (
            alt.Chart(top_tags)
            .mark_bar()
            .encode(
                x=alt.X("count:Q", title="出現次數"),
                y=alt.Y("tag:N", title="標籤", sort="-x"),
                color=alt.Color("count:Q", scale=alt.Scale(scheme="blues"), legend=None),
                tooltip=["tag:N", "count:Q"]
            )
        ).properties(
            width="container",
            height=360,
            title="熱門語意標籤 Top 15"
        )

        st.altair_chart(tag_chart, use_container_width=True)
    else:
        st.info("語意分析未產出標籤，無法顯示熱門標籤圖。")


# =========================
# 5. Streamlit 介面
# =========================

def main():
    st.set_page_config(
        page_title="香港電影粵語評論分析助手",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🎬 香港影片／電影粵語評論分析助手")
    st.markdown(
        "使用 YouTube Data API 與 DeepSeek (OpenAI 相容介面) 分析目標影片的粵語留言。"
        "具備 API 配額最佳化與快取機制。"
    )

    with st.sidebar:
        st.header("🔑 API 設定")
        yt_api_key = st.text_input(
            "YouTube Data API 金鑰",
            type="password",
            help="可至 Google Cloud Console 建立 API Key。"
        )
        deepseek_api_key = st.text_input(
            "DeepSeek API 金鑰",
            type="password",
            help="選填。如未提供，將跳過語意分析。"
        )

        st.header("🎯 分析參數")
        movie_title = st.text_input("搜尋影片／電影名稱", value="梅艷芳")
        today = date.today()
        default_start = today - timedelta(days=30)

        start_date = st.date_input(
            "起始日期 (香港時間)",
            value=default_start
        )
        end_date = st.date_input(
            "結束日期 (香港時間)",
            value=today
        )

        max_videos_per_keyword = st.number_input(
            "每個關鍵字最多影片數",
            min_value=5,
            max_value=200,
            value=30,
            step=5
        )
        max_comments_per_video = st.number_input(
            "每部影片最多留言數",
            min_value=10,
            max_value=500,
            value=80,
            step=10
        )
        sample_size = st.number_input(
            "分析留言上限 (0 = 不限制)",
            min_value=0,
            max_value=2000,
            value=0,
            step=50
        )
        cantonese_threshold = st.slider(
            "粵語分數門檻",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.5
        )
        relax_trad_filter = st.checkbox(
            "允許未知／混合字體 (zh-unkn)",
            value=True
        )
        auto_relax_threshold = st.checkbox(
            "自動放寬粵語門檻以達指定樣本數",
            value=True
        )
        target_min_cantonese = st.number_input(
            "目標最少粵語留言數",
            min_value=50,
            max_value=2000,
            value=300,
            step=50
        )
        prefer_hk_videos = st.checkbox(
            "優先分析香港傾向影片",
            value=True
        )

        start_date_str = start_date.isoformat()
        end_date_str = end_date.isoformat()

        if end_date < start_date:
            st.error("結束日期必須晚於或等於起始日期。")

        st.markdown("---")
        run_button = st.button("🚀 開始分析")

    if run_button:
        if not yt_api_key:
            st.error("請輸入 YouTube Data API 金鑰。")
            st.stop()

        with st.spinner("🔍 正在搜尋相關影片並擷取留言..."):
            df_result, error = movie_comment_analysis(
                movie_title=movie_title.strip(),
                start_date=start_date_str,
                end_date=end_date_str,
                yt_api_key=yt_api_key.strip(),
                deepseek_api_key=deepseek_api_key.strip() if deepseek_api_key else None,
                max_videos_per_keyword=int(max_videos_per_keyword),
                max_comments_per_video=int(max_comments_per_video),
                sample_size=int(sample_size) if sample_size else None,
                relax_trad_filter=relax_trad_filter,
                cantonese_threshold=float(cantonese_threshold),
                auto_relax_threshold=auto_relax_threshold,
                target_min_cantonese=int(target_min_cantonese),
                prefer_hk_videos=prefer_hk_videos
            )

        if error:
            st.error(error)
            return

        st.success(f"共分析 {len(df_result)} 則留言。")

        render_visualizations(df_result)

        st.markdown("### 📊 分析結果一覽")
        st.dataframe(
            df_result[
                [
                    "video_title",
                    "video_hk_score",
                    "cantonese_score",
                    "analysis_sentiment",
                    "analysis_sentiment_confidence",
                    "analysis_tags",
                    "analysis_summary",
                    "comment_text",
                    "comment_like_count",
                    "comment_published_at_hk",
                    "video_url"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")
        st.markdown("### 📥 下載結果")
        csv_bytes = df_result.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="下載 CSV",
            data=csv_bytes,
            file_name=f"{movie_title}_comment_analysis.csv",
            mime="text/csv"
        )

        st.markdown("### 🧾 參數總結")
        st.json({
            "movie_title": movie_title,
            "date_range": [start_date_str, end_date_str],
            "max_videos_per_keyword": max_videos_per_keyword,
            "max_comments_per_video": max_comments_per_video,
            "sample_size": sample_size,
            "cantonese_threshold_final": df_result["cantonese_threshold_applied"].iloc[0],
            "prefer_hk_videos": prefer_hk_videos,
            "deepseek_enabled": bool(deepseek_api_key)
        })


if __name__ == "__main__":
    main()
