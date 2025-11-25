# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import time
import asyncio
import openai
import re
import json
from opencc import OpenCC
from googleapiclient.discovery import build

# =========================
# 0. 快取與工具函式
# =========================

# 快取過期時間設定
CACHE_TTL_SEARCH = 3600          # 1 小時：搜尋結果
CACHE_TTL_RELEVANCE = 86400      # 24 小時：AI 相關性判斷結果（影片標題不變，判斷結果就不變）
CACHE_TTL_CHANNEL = 86400        # 24 小時：頻道國家資訊
CACHE_TTL_COMMENTS = 900         # 15 分鐘：留言清單

def _get_cached_value(cache_name: str, key, ttl_seconds: int):
    """從 st.session_state 獲取快取，若過期則返回 None"""
    if cache_name not in st.session_state:
        st.session_state[cache_name] = {}
    
    entry = st.session_state[cache_name].get(key)
    if entry:
        if (time.time() - entry["ts"]) <= ttl_seconds:
            return entry["value"]
        else:
            del st.session_state[cache_name][key]
    return None

def _set_cached_value(cache_name: str, key, value):
    """寫入快取到 st.session_state"""
    if cache_name not in st.session_state:
        st.session_state[cache_name] = {}
    st.session_state[cache_name][key] = {
        "value": value,
        "ts": time.time()
    }

# =========================
# 1. 語言與關鍵字工具
# =========================

def generate_search_queries(movie_title: str):
    zh_terms = [
        "影評", "評論", "評價", "點評", "解析", "分析", "觀後感",
        "無雷", "有雷", "討論", "好唔好睇", "預告", "花絮", "片段", "首映", "幕後",
        "香港", "香港上映", "香港首映", "香港反應", "戲院 反應", "院線", "街訪",
        "粵語", "廣東話", "粵語配音", "粵配", "港版", "港產"
    ]
    en_terms = [
        "review", "reaction", "ending explained", "analysis", "explained",
        "behind the scenes", "bts", "premiere", "interview", "press conference",
        "hong kong", "hk reaction", "hk audience"
    ]

    loose = [f"{movie_title}"]
    loose += [f"{movie_title} {t}" for t in zh_terms]
    loose += [f"{movie_title} {t}" for t in en_terms]

    tight = [
        f"\"{movie_title}\"",
        f"\"{movie_title}\" 影評",
        f"\"{movie_title}\" 評論",
        f"\"{movie_title}\" 解析",
        f"\"{movie_title}\" review",
        f"\"{movie_title}\" reaction",
        f"\"{movie_title}\" 香港",
        f"\"{movie_title}\" 粵語",
        f"\"{movie_title}\" 廣東話",
    ]

    seen = set()
    queries = []
    for q in loose + tight:
        if q not in seen:
            queries.append(q)
            seen.add(q)
    return queries

CANTONESE_CHAR_TOKENS = {
    "唔": 1.0, "冇": 1.6, "咗": 1.6, "嘅": 1.6, "啲": 1.2, "嗰": 1.2, "佢": 1.0,
    "喺": 1.6, "嚟": 1.6, "咪": 1.2, "啱": 1.2, "掂": 1.2, "靚": 1.2, "曳": 1.2,
    "攰": 1.2, "咁": 1.0, "噉": 1.0, "得": 0.6, "吖": 0.8, "冧": 1.0, "撚": 1.2,
    "仆": 1.2, "屌": 1.2, "嗮": 1.0, "畀": 0.8, "揸": 1.0, "腎": 0.0
}
CANTONESE_PARTICLES = ["啦", "囉", "喎", "咩", "呢", "呀", "嘛", "喇"]
CANTONESE_PHRASES = {
    "好唔好睇": 2.0, "做咩": 1.6, "點解": 1.2, "咩料": 1.6, "算啦": 1.2,
    "得啦": 1.2, "正喎": 1.2, "幾好睇": 1.6, "幾正": 1.2, "好正": 1.0,
    "有啲": 0.8, "嗰啲": 1.2, "呢啲": 1.2, "講真": 0.8, "好似": 0.5
}
ROMANIZATION_RE = re.compile(r"(?i)(?<![A-Za-z])(la|lor|wor|leh|meh|mah|ga|wo|ar)(?=[\s\W]|$)")

def count_chars(text: str):
    counts = {
        "cjk": 0, "hiragana": 0, "katakana": 0, "half_katakana": 0,
        "hangul": 0, "latin": 0, "digits": 0, "other": 0
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
    m = min(len(a), len(b))
    return sum(1 for i in range(m) if a[i] != b[i]) + abs(len(a) - len(b))

def classify_zh_trad_simp(text: str, cc_t2s: OpenCC, cc_s2t: OpenCC):
    if not isinstance(text, str) or len(text.strip()) < 2:
        return "other"
    counts = count_chars(text)
    kana = counts["hiragana"] + counts["katakana"] + counts["half_katakana"]
    cjk = counts["cjk"]
    if kana >= 2 and kana / max(1, (cjk + kana)) >= 0.10:
        return "ja"
    if cjk < 1:
        return "other"
    t2s = cc_t2s.convert(text)
    s2t = cc_s2t.convert(text)
    ct2s = diff_chars(text, t2s)
    cs2t = diff_chars(text, s2t)
    threshold = max(1, int(0.05 * cjk))
    if ct2s > cs2t + threshold:
        return "zh-Hant"
    elif cs2t > ct2s + threshold:
        return "zh-Hans"
    else:
        return "zh-unkn"

def score_cantonese(text: str) -> float:
    if not isinstance(text, str) or not text:
        return 0.0
    score = 0.0
    for phrase, w in CANTONESE_PHRASES.items():
        cnt = text.count(phrase)
        if cnt:
            score += cnt * w
    for ch, w in CANTONESE_CHAR_TOKENS.items():
        cnt = text.count(ch)
        if cnt:
            score += cnt * w
    end_slice = text[-8:] if len(text) > 8 else text
    for p in CANTONESE_PARTICLES:
        cnt = text.count(p)
        if cnt:
            score += cnt * 0.6
        if p in end_slice:
            score += 0.4
    roman_hits = ROMANIZATION_RE.findall(text)
    if roman_hits:
        score += len(roman_hits) * 0.8
    return score

# =========================
# 2. YouTube 搜尋 (含快取與總量控制)
# =========================

def search_youtube_videos(
    keywords,
    youtube_client,
    max_per_keyword,
    start_date,
    end_date,
    add_language_bias=True,
    region_bias=True,
    max_total_videos=150
):
    all_video_ids = set()
    video_meta = {}
    status_text = st.empty()

    for idx, query in enumerate(keywords):
        if len(all_video_ids) >= max_total_videos:
            status_text.info(f"已達到搜尋上限 ({max_total_videos} 部)，停止後續搜尋。")
            break

        cache_key = f"{query}_{start_date}_{end_date}_{max_per_keyword}_{add_language_bias}_{region_bias}"
        cached_data = _get_cached_value("search_cache", cache_key, CACHE_TTL_SEARCH)

        query_records = []

        if cached_data is not None:
            query_records = cached_data
        else:
            collected_records = []
            collected_ids_for_query = set()
            
            for order in ["relevance", "viewCount"]:
                if len(collected_records) >= max_per_keyword:
                    break

                try:
                    request = youtube_client.search().list(
                        q=query,
                        part="id,snippet",
                        type="video",
                        maxResults=50,
                        publishedAfter=f"{start_date}T00:00:00Z",
                        publishedBefore=f"{end_date}T23:59:59Z",
                        order=order,
                        safeSearch="none",
                        **({"relevanceLanguage": "zh-Hant"} if add_language_bias else {}),
                        **({"regionCode": "HK"} if region_bias else {})
                    )
                    
                    while request and len(collected_records) < max_per_keyword:
                        response = request.execute()
                        for item in response.get("items", []):
                            vid = item["id"]["videoId"]
                            if vid in collected_ids_for_query:
                                continue
                            
                            collected_ids_for_query.add(vid)
                            snip = item.get("snippet", {})
                            record = {
                                "video_id": vid,
                                "title": snip.get("title", ""),
                                "channelTitle": snip.get("channelTitle", ""),
                                "publishedAt": snip.get("publishedAt", "")
                            }
                            collected_records.append(record)
                        
                        if len(collected_records) >= max_per_keyword:
                            break
                            
                        request = youtube_client.search().list_next(request, response)
                        time.sleep(0.1)
                except Exception as e:
                    st.warning(f"搜尋 '{query}' 時發生錯誤: {e}")
                    continue
            
            _set_cached_value("search_cache", cache_key, collected_records)
            query_records = collected_records

        for record in query_records:
            vid = record["video_id"]
            if vid not in all_video_ids:
                all_video_ids.add(vid)
                if vid not in video_meta:
                    video_meta[vid] = {
                        "title": record["title"],
                        "channelTitle": record["channelTitle"],
                        "publishedAt": record["publishedAt"]
                    }
            if len(all_video_ids) >= max_total_videos:
                break
    
    status_text.empty()
    return list(all_video_ids), video_meta

# =========================
# NEW: AI 影片相關性過濾
# =========================

async def check_relevance_batch_async(movie_title, batch_videos, deepseek_client):
    """
    使用 DeepSeek 批量判斷影片是否與電影相關。
    batch_videos: list of {"id": vid, "title": title, "channel": channel}
    """
    if not batch_videos:
        return []

    prompt_items = []
    for v in batch_videos:
        prompt_items.append(f"ID: {v['id']}\nTitle: {v['title']}\nChannel: {v['channel']}")
    
    prompt_text = "\n---\n".join(prompt_items)

    system_prompt = (
        f"You are a data cleaner. The user is analyzing the movie '{movie_title}'. "
        "Below is a list of YouTube video titles found by search. "
        "Identify which videos are actually discussing this specific movie (reviews, reactions, news, clips, interviews). "
        "Exclude videos that are clearly unrelated (e.g., generic news, other movies, music videos not related to the film, or completely different topics). "
        "Return a JSON object where keys are the Video IDs and values are boolean true (relevant) or false (irrelevant). "
        "Example: {\"vid123\": true, \"vid456\": false}"
    )

    try:
        response = await deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        data = json.loads(response.choices[0].message.content)
        # 提取為 True 的 ID
        valid_ids = [vid for vid, is_relevant in data.items() if is_relevant is True]
        return valid_ids
    except Exception as e:
        # 如果 API 失敗，為了安全起見，默認保留（Fail Open），以免丟失數據
        # 或者也可以選擇僅在失敗時保留包含電影名的
        print(f"Relevance check failed: {e}")
        return [v['id'] for v in batch_videos]

async def filter_videos_by_relevance(movie_title, video_ids, video_meta, deepseek_client):
    """
    主入口：過濾影片列表
    """
    # 1. 檢查快取
    to_check = []
    valid_ids = set()
    
    for vid in video_ids:
        cached_res = _get_cached_value("relevance_cache", f"{movie_title}_{vid}", CACHE_TTL_RELEVANCE)
        if cached_res is not None:
            if cached_res:
                valid_ids.add(vid)
        else:
            meta = video_meta.get(vid, {})
            to_check.append({
                "id": vid,
                "title": meta.get("title", ""),
                "channel": meta.get("channelTitle", "")
            })
    
    # 2. 批量送去 AI 檢查
    if to_check:
        batch_size = 20
        tasks = []
        
        # 分批建立異步任務
        for i in range(0, len(to_check), batch_size):
            batch = to_check[i:i+batch_size]
            tasks.append(check_relevance_batch_async(movie_title, batch, deepseek_client))
        
        # 執行任務
        progress_text = st.empty()
        progress_text.info(f"正在使用 AI 過濾 {len(to_check)} 部影片的相關性...")
        
        results = await asyncio.gather(*tasks)
        
        # 整合結果並寫入快取
        for batch_idx, relevant_list in enumerate(results):
            batch_input = to_check[batch_idx*batch_size : (batch_idx+1)*batch_size]
            # 建立一個 lookup set
            rel_set = set(relevant_list)
            
            for item in batch_input:
                vid = item["id"]
                is_rel = vid in rel_set
                if is_rel:
                    valid_ids.add(vid)
                
                # 寫入快取
                _set_cached_value("relevance_cache", f"{movie_title}_{vid}", is_rel)
        
        progress_text.empty()

    return list(valid_ids)

# =========================
# 3. 獲取詳情與留言
# =========================

def fetch_video_and_channel_details(video_ids, youtube_client):
    video_extra = {}
    channel_ids = set()

    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i:i+50]
        try:
            resp = youtube_client.videos().list(
                part="snippet,contentDetails",
                id=",".join(chunk)
            ).execute()
            for item in resp.get("items", []):
                vid = item.get("id")
                snip = item.get("snippet", {}) or {}
                ch = snip.get("channelId")
                video_extra[vid] = {
                    "channelId": ch,
                    "defaultLanguage": (snip.get("defaultLanguage") or ""),
                    "defaultAudioLanguage": (snip.get("defaultAudioLanguage") or ""),
                    "tags": snip.get("tags", [])
                }
                if ch:
                    channel_ids.add(ch)
        except Exception as e:
            st.warning(f"videos.list 取資料時發生錯誤: {e}")

    channel_country = {}
    channels_to_fetch = []

    for cid in channel_ids:
        cached_country = _get_cached_value("channel_cache", cid, CACHE_TTL_CHANNEL)
        if cached_country is not None:
            channel_country[cid] = cached_country
        else:
            channels_to_fetch.append(cid)
    
    if channels_to_fetch:
        for i in range(0, len(channels_to_fetch), 50):
            chunk = channels_to_fetch[i:i+50]
            try:
                resp = youtube_client.channels().list(
                    part="brandingSettings",
                    id=",".join(chunk)
                ).execute()
                for item in resp.get("items", []):
                    cid = item.get("id")
                    brand = (item.get("brandingSettings", {}) or {}).get("channel", {}) or {}
                    country = brand.get("country")
                    channel_country[cid] = country
                    _set_cached_value("channel_cache", cid, country)
            except Exception as e:
                st.warning(f"channels.list 取資料時發生錯誤: {e}")

    return video_extra, channel_country

def compute_hk_video_score(video_id, video_meta, video_extra, channel_country_map):
    meta = video_meta.get(video_id, {}) or {}
    ext = video_extra.get(video_id, {}) or {}
    title = meta.get("title", "") or ""
    tags = " ".join(ext.get("tags", []) or [])
    ch = ext.get("channelId")
    default_audio = (ext.get("defaultAudioLanguage") or "").lower()
    country = channel_country_map.get(ch)

    score = 0
    if country == "HK": score += 3
    if default_audio in ("yue", "zh-hk", "zh-yue", "zh-hant-hk"): score += 3
    elif default_audio.startswith("zh"): score += 1
    
    if any(tok in title for tok in ["粵語", "廣東話", "粵配", "粵語配音"]): score += 3
    if any(tok in title for tok in ["香港", "港版", "香港觀眾", "香港反應", "香港首映", "香港上映"]): score += 2
    if ("HK" in title) or ("Hong Kong" in title): score += 1
    if any(tok in tags for tok in ["粵語", "廣東話", "香港", "HK"]): score += 2
    return score

def get_all_comments(
    video_ids, youtube_client, max_per_video, 
    video_meta=None, hk_score_map=None, video_extra=None, channel_country_map=None,
    max_total_comments=2000
):
    video_meta = video_meta or {}
    hk_score_map = hk_score_map or {}
    video_extra = video_extra or {}
    channel_country_map = channel_country_map or {}

    all_comments = []
    total_videos = len(video_ids)
    progress_bar = st.progress(0, text="抓取 YouTube 留言中...")
    
    total_fetched_count = 0

    for i, video_id in enumerate(video_ids):
        if total_fetched_count >= max_total_comments:
            break

        cache_key = f"{video_id}_{max_per_video}"
        cached_comments = _get_cached_value("comments_cache", cache_key, CACHE_TTL_COMMENTS)
        
        current_video_comments = []

        if cached_comments is not None:
            current_video_comments = cached_comments
        else:
            try:
                request = youtube_client.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    textFormat="plainText",
                    order="time",
                    maxResults=100
                )
                fetched_for_video = 0
                raw_records = []
                
                while request and fetched_for_video < max_per_video:
                    if total_fetched_count + fetched_for_video >= max_total_comments:
                        break

                    response = request.execute()
                    for item in response.get("items", []):
                        if fetched_for_video >= max_per_video:
                            break
                        comment = item["snippet"]["topLevelComment"]["snippet"]
                        
                        record = {
                            "textDisplay": comment.get("textDisplay", ""),
                            "publishedAt": comment.get("publishedAt", ""),
                            "likeCount": comment.get("likeCount", 0)
                        }
                        raw_records.append(record)
                        fetched_for_video += 1
                    
                    request = youtube_client.commentThreads().list_next(request, response)
                    if request and fetched_for_video < max_per_video:
                        time.sleep(0.1)
                
                if raw_records:
                    _set_cached_value("comments_cache", cache_key, raw_records)
                current_video_comments = raw_records

            except Exception:
                pass

        ch_id = (video_extra.get(video_id, {}) or {}).get("channelId")
        ch_country = channel_country_map.get(ch_id) if ch_id else None
        def_audio = (video_extra.get(video_id, {}) or {}).get("defaultAudioLanguage", "")
        v_title = video_meta.get(video_id, {}).get("title", "")
        v_score = hk_score_map.get(video_id, 0)

        for raw in current_video_comments:
            all_comments.append({
                "video_id": video_id,
                "video_title": v_title,
                "video_url": f"https://www.youtube.com/watch?v={video_id}",
                "video_hk_score": v_score,
                "video_channel_id": ch_id,
                "video_channel_country": ch_country,
                "video_default_audio_lang": def_audio,
                "comment_text": raw["textDisplay"],
                "published_at": raw["publishedAt"],
                "like_count": raw["likeCount"]
            })
        
        total_fetched_count += len(current_video_comments)

        progress_bar.progress(
            (i + 1) / max(1, total_videos),
            text=f"抓取 YouTube 留言中... ({min(i+1, total_videos)}/{total_videos} 部影片, 已抓 {total_fetched_count} 則)"
        )

    progress_bar.empty()
    return pd.DataFrame(all_comments)

# =========================
# 4. DeepSeek AI 情感分析
# =========================

async def analyze_comment_deepseek_async(comment_text, deepseek_client, semaphore, max_retries=3):
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
# 5. 主流程
# =========================

def movie_comment_analysis(
    movie_title, start_date, end_date,
    yt_api_key, deepseek_api_key,
    max_videos_per_keyword=30, max_comments_per_video=50, sample_size=None,
    relax_trad_filter=True,
    cantonese_threshold=2.0,
    auto_relax_threshold=True,
    target_min_cantonese=300,
    prefer_hk_videos=True
):
    target_sample = sample_size if sample_size and sample_size > 0 else 1000
    GLOBAL_MAX_COMMENTS = max(2000, target_sample * 4) 
    GLOBAL_MAX_VIDEOS = 150

    SEARCH_KEYWORDS = generate_search_queries(movie_title)

    youtube_client = build("youtube", "v3", developerKey=yt_api_key)
    deepseek_client = openai.AsyncOpenAI(
        api_key=deepseek_api_key,
        base_url="https://api.deepseek.com/v1"
    )

    # 1) 搜尋
    video_ids, video_meta = search_youtube_videos(
        SEARCH_KEYWORDS, youtube_client, max_videos_per_keyword, start_date, end_date,
        add_language_bias=True, region_bias=True,
        max_total_videos=GLOBAL_MAX_VIDEOS
    )
    if not video_ids:
        return None, "找不到相關影片。"
    
    st.info(f"初步搜尋到 {len(video_ids)} 部影片，正在進行 AI 相關性過濾...")

    # 2) NEW: AI 相關性過濾 (使用 DeepSeek)
    relevant_video_ids = asyncio.run(filter_videos_by_relevance(movie_title, video_ids, video_meta, deepseek_client))
    
    removed_count = len(video_ids) - len(relevant_video_ids)
    if removed_count > 0:
        st.warning(f"AI 已過濾掉 {removed_count} 部與「{movie_title}」不相關的影片，保留 {len(relevant_video_ids)} 部進行分析。")
    else:
        st.info("所有搜尋到的影片均被判定為相關。")

    if not relevant_video_ids:
        return None, "AI 過濾後沒有剩餘相關影片，請嘗試更換關鍵字或檢查電影名稱。"

    # 3) 獲取詳細資料 (只針對相關影片)
    video_extra, channel_country_map = fetch_video_and_channel_details(relevant_video_ids, youtube_client)

    # 4) 影片香港傾向排序
    hk_score_map = {vid: compute_hk_video_score(vid, video_meta, video_extra, channel_country_map) for vid in relevant_video_ids}
    video_ids_sorted = sorted(relevant_video_ids, key=lambda v: hk_score_map.get(v, 0), reverse=True) if prefer_hk_videos else relevant_video_ids

    # 5) 抓取留言
    df_comments = get_all_comments(
        video_ids_sorted, youtube_client, max_comments_per_video,
        video_meta=video_meta, hk_score_map=hk_score_map,
        video_extra=video_extra, channel_country_map=channel_country_map,
        max_total_comments=GLOBAL_MAX_COMMENTS
    )
    if df_comments.empty:
        return None, "找不到任何留言。"

    # 6) 語言過濾
    st.info(f"已抓取 {len(df_comments)} 則原始留言，現開始語言與粵語篩選...")

    cc_t2s = OpenCC("t2s")
    cc_s2t = OpenCC("s2t")

    def lang_pred(text):
        return classify_zh_trad_simp(text, cc_t2s, cc_s2t)

    df_comments["lang_pred"] = df_comments["comment_text"].apply(lang_pred)
    df_comments = df_comments[~df_comments["lang_pred"].isin(["ja", "other", "zh-Hans"])].reset_index(drop=True)

    if relax_trad_filter:
        df_comments = df_comments[df_comments["lang_pred"].isin(["zh-Hant", "zh-unkn"])].reset_index(drop=True)
    else:
        df_comments = df_comments[df_comments["lang_pred"] == "zh-Hant"].reset_index(drop=True)

    if df_comments.empty:
        return None, "在抓取的留言中沒有符合基本語言條件的內容。"

    # 粵語分數
    df_comments["cantonese_score"] = df_comments["comment_text"].apply(score_cantonese)

    # 7) 粵語門檻 + 自動放寬
    thr = float(cantonese_threshold)
    def filt(t): return t >= thr
    df_filtered = df_comments[df_comments["cantonese_score"].apply(filt)].reset_index(drop=True)

    if auto_relax_threshold and len(df_filtered) < target_min_cantonese:
        new_thr = thr
        while len(df_filtered) < target_min_cantonese and new_thr > 0.5:
            new_thr = round(new_thr - 0.5, 2)
            df_filtered = df_comments[df_comments["cantonese_score"] >= new_thr].reset_index(drop=True)
        if new_thr != thr:
            st.info(f"自動放寬粵語分數門檻：{thr} ➜ {new_thr}（目前符合條件留言：{len(df_filtered)}）")
            thr = new_thr

    st.info(f"語言與粵語篩選後剩下 {len(df_filtered)} 則留言（門檻={thr}）。")
    if df_filtered.empty:
        return None, "粵語篩選後樣本為 0，請調低門檻或延長時間範圍。"

    # 8) 時區與日期篩選
    df_filtered["published_at"] = pd.to_datetime(df_filtered["published_at"], utc=True, errors="coerce")
    df_filtered["published_at_hk"] = df_filtered["published_at"].dt.tz_convert("Asia/Hong_Kong")

    start_dt = pd.to_datetime(start_date).tz_localize("Asia/Hong_Kong")
    end_dt = pd.to_datetime(end_date).tz_localize("Asia/Hong_Kong") + timedelta(days=1)
    mask_date = (df_filtered["published_at_hk"] >= start_dt) & (df_filtered["published_at_hk"] < end_dt)
    df_filtered = df_filtered.loc[mask_date].reset_index(drop=True)
    if df_filtered.empty:
        return None, "在指定日期範圍內沒有符合粵語條件的留言。"

    # 9) 取樣控制
    if sample_size and 0 < sample_size < len(df_filtered):
        df_analyze = df_filtered.sample(n=sample_size, random_state=42)
    else:
        df_analyze = df_filtered

    st.info(f"準備對 {len(df_analyze)} 則留言進行高速並發分析...")

    # 10) DeepSeek 分析
    analysis_results = asyncio.run(run_all_analyses(df_analyze, deepseek_client))
    analysis_df = pd.DataFrame(analysis_results)
    final_df = pd.concat([df_analyze.reset_index(drop=True), analysis_df], axis=1)

    final_df["published_at"] = pd.to_datetime(final_df["published_at"])
    return final_df, None

# =========================
# 6. Streamlit UI
# =========================

st.set_page_config(page_title="YouTube 電影評論 AI 分析（香港粵語優先）", layout="wide")
st.title("🎬 YouTube 電影評論 AI 情感分析（香港粵語優先）")

with st.expander("使用說明"):
    st.markdown("""
    1.  輸入電影的中文全名、分析時間範圍及所需的 API 金鑰。
    2.  本工具會偏向抓取香港地區的影片與留言，並用粵語特徵打分過濾。
    3.  **優化版**：
        *   **智能快取**：重複查詢不消耗 YouTube 配額。
        *   **AI 相關性過濾**：使用 DeepSeek 自動剔除標題不相關的影片，確保分析精準度並節省 YouTube 留言抓取配額。
    4.  分析完成後，提供可視化與 CSV 下載。
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
max_videos = st.slider("每個關鍵字的最大影片搜尋數", 5, 80, 30, help="提高可增加覆蓋，但會增加 YouTube API 配額消耗。")
max_comments = st.slider("每部影片的最大留言抓取數", 10, 200, 80, help="數量越多，分析結果越全面，但 DeepSeek API 成本越高。")
sample_size = st.number_input("分析留言數量上限 (0=全量)", 0, 5000, 500)

relax_trad_filter = st.checkbox("放寬繁體判定（允許難分的中文留言）", value=True)
prefer_hk_videos = st.checkbox("優先抓取更可能來自香港/粵語的影片（排序加權）", value=True)

cantonese_threshold = st.slider("粵語分數門檻", 0.5, 6.0, 2.0, 0.5, help="分數越高越嚴格，2.0 是較穩健的門檻。")
auto_relax_threshold = st.checkbox("自動放寬門檻以達到目標樣本量", value=True)
target_min_cantonese = st.number_input("目標最少粵語評論數（啟用自動放寬時生效）", 50, 5000, 300)

if st.button("🚀 開始分析"):
    if not all([movie_title, yt_api_key, deepseek_api_key]):
        st.warning("請填寫電影名稱和兩個 API 金鑰。")
    else:
        with st.spinner("AI 高速分析中... 請稍候..."):
            df_result, err = movie_comment_analysis(
                movie_title, str(start_date), str(end_date),
                yt_api_key, deepseek_api_key,
                max_videos, max_comments, sample_size,
                relax_trad_filter=relax_trad_filter,
                cantonese_threshold=cantonese_threshold,
                auto_relax_threshold=auto_relax_threshold,
                target_min_cantonese=target_min_cantonese,
                prefer_hk_videos=prefer_hk_videos
            )

        if err:
            st.error(err)
        else:
            st.success("分析完成！")
            st.dataframe(df_result.head(20), use_container_width=True)

            st.header("📊 可視化分析結果")

            sentiments_order = ['Positive', 'Negative', 'Neutral', 'Invalid', 'Error']
            colors_map = {
                'Positive': '#5cb85c', 'Negative': '#d9534f', 'Neutral': '#f0ad4e',
                'Invalid': '#cccccc', 'Error': '#888888'
            }

            # 1. 情感分佈
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
                    color_discrete_map=colors_map
                )
                st.plotly_chart(fig1, use_container_width=True)
            else:
                st.info("No sentiment data available for pie chart.")

            # 2. 每日情感趨勢
            st.subheader("2. Daily Sentiment Trend")
            if 'published_at_hk' in df_result.columns:
                df_result['date'] = df_result['published_at_hk'].dt.date
            else:
                df_result['date'] = pd.to_datetime(df_result['published_at'], utc=True).dt.tz_convert('Asia/Hong_Kong').dt.date

            daily = df_result.groupby(['date', 'sentiment']).size().unstack().fillna(0)
            daily = daily.reindex(columns=sentiments_order).dropna(axis=1, how='all')
            if not daily.empty:
                daily_long = daily.reset_index().melt(id_vars='date', var_name='sentiment', value_name='count')
                fig_line = px.line(
                    daily_long, x='date', y='count', color='sentiment',
                    title='Daily Comment Volume Trend by Sentiment',
                    labels={'date': 'Date', 'count': 'Number of Comments', 'sentiment': 'Sentiment'},
                    color_discrete_map=colors_map
                )
                st.plotly_chart(fig_line, use_container_width=True)

                fig_bar = px.bar(
                    daily_long, x='date', y='count', color='sentiment',
                    title='Daily Comment Volume by Sentiment (Stacked)',
                    labels={'date': 'Date', 'count': 'Number of Comments', 'sentiment': 'Sentiment'},
                    color_discrete_map=colors_map,
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

            # 4. 下載分析明細
            st.subheader("4. 下載分析明細")
            csv = df_result.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                "📥 下載全部分析明細 (CSV)",
                csv,
                file_name=f"{movie_title}_hk_cantonese_analysis.csv",
                mime='text/csv'
            )
