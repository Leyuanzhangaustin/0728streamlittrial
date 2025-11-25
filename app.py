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

CACHE_TTL_SEARCH = 3600
CACHE_TTL_RELEVANCE = 86400
CACHE_TTL_CHANNEL = 86400
CACHE_TTL_COMMENTS = 900

def _get_cached_value(cache_name: str, key, ttl_seconds: int):
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
    if cache_name not in st.session_state:
        st.session_state[cache_name] = {}
    st.session_state[cache_name][key] = {
        "value": value,
        "ts": time.time()
    }

# =========================
# 1. 語言與關鍵字工具 (已更新字典)
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

# 更新：增加更多口語變體和俚語
CANTONESE_CHAR_TOKENS = {
    "唔": 1.0, "冇": 1.6, "咗": 1.6, "嘅": 1.6, "啲": 1.2, "嗰": 1.2, "佢": 1.0,
    "喺": 1.6, "嚟": 1.6, "咪": 1.2, "啱": 1.2, "掂": 1.2, "靚": 1.2, "曳": 1.2,
    "攰": 1.2, "咁": 1.0, "噉": 1.0, "得": 0.6, "吖": 0.8, "冧": 1.0, "撚": 1.2,
    "仆": 1.2, "屌": 1.2, "嗮": 1.0, "畀": 0.8, "揸": 1.0, "腎": 0.0,
    # 新增/調整
    "系": 0.5,  # 很多人打錯字 "系" 代替 "係"，雖然簡體也有，但在繁體環境下出現通常是粵語
    "係": 1.5,  # 核心詞
    "9": 0.5,   # 數字俚語 (鳩/狗)
    "7": 0.5,   # 數字俚語 (柒)
    "6": 0.3,   # 數字俚語 (陸/碌)
    "亞": 0.5,  # 亞媽, 亞哥 (阿的異體)
    "野": 0.5,  # 嘢的異體
    "既": 0.5,  # 嘅的異體
    "左": 0.5,  # 咗的異體
    "d": 0.8, "D": 0.8, # 啲的代號
}

CANTONESE_PARTICLES = ["啦", "囉", "喎", "咩", "呢", "呀", "嘛", "喇", "杰", "姐", "噃"]
CANTONESE_PHRASES = {
    "好唔好睇": 2.0, "做咩": 1.6, "點解": 1.6, "咩料": 1.6, "算啦": 1.2,
    "得啦": 1.2, "正喎": 1.2, "幾好睇": 1.6, "幾正": 1.2, "好正": 1.0,
    "有啲": 0.8, "嗰啲": 1.2, "呢啲": 1.2, "講真": 0.8, "好似": 0.5,
    "多9余": 2.0, "多餘": 0.5, "真系": 1.0, "真係": 1.5, "打風": 1.0
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
    if not isinstance(text, str) or len(text.strip()) < 1:
        return "other"
    counts = count_chars(text)
    
    # 判斷是否主要為英文
    total_chars = len(text.strip())
    if counts["latin"] / max(1, total_chars) > 0.7:
        return "en"

    kana = counts["hiragana"] + counts["katakana"] + counts["half_katakana"]
    cjk = counts["cjk"]
    
    if kana >= 2 and kana / max(1, (cjk + kana)) >= 0.10:
        return "ja"
    if cjk < 1:
        # 如果沒有中文字，但也不是英文，歸類為其他
        return "other" if counts["latin"] == 0 else "en"

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
    text_lower = text.lower() # 處理 d/D
    
    for phrase, w in CANTONESE_PHRASES.items():
        if phrase in text: # 區分大小寫的匹配 (中文)
            score += text.count(phrase) * w
            
    for ch, w in CANTONESE_CHAR_TOKENS.items():
        # 對於英文代號 d/D，我們用 lower 檢查
        if ch in ['d', 'D']:
            cnt = text_lower.count('d')
            # 簡單防止單詞中的 d (如 and, good) 被誤判，這裡只是一個粗略過濾
            # 更好的方法是用 regex，但這裡從簡，假設 d 旁邊有中文或空格
            score += cnt * w * 0.5 
        else:
            cnt = text.count(ch)
            if cnt:
                score += cnt * w

    end_slice = text[-8:] if len(text) > 8 else text
    for p in CANTONESE_PARTICLES:
        if p in end_slice:
            score += 0.4
        elif p in text:
            score += 0.2 # 非結尾語氣詞權重較低

    roman_hits = ROMANIZATION_RE.findall(text)
    if roman_hits:
        score += len(roman_hits) * 0.8
    return score

# =========================
# 2. YouTube 搜尋
# =========================

def search_youtube_videos(keywords, youtube_client, max_per_keyword, start_date, end_date, add_language_bias=True, region_bias=True, max_total_videos=150):
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
                if len(collected_records) >= max_per_keyword: break
                try:
                    request = youtube_client.search().list(
                        q=query, part="id,snippet", type="video", maxResults=50,
                        publishedAfter=f"{start_date}T00:00:00Z", publishedBefore=f"{end_date}T23:59:59Z",
                        order=order, safeSearch="none",
                        **({"relevanceLanguage": "zh-Hant"} if add_language_bias else {}),
                        **({"regionCode": "HK"} if region_bias else {})
                    )
                    while request and len(collected_records) < max_per_keyword:
                        response = request.execute()
                        for item in response.get("items", []):
                            vid = item["id"]["videoId"]
                            if vid in collected_ids_for_query: continue
                            collected_ids_for_query.add(vid)
                            snip = item.get("snippet", {})
                            collected_records.append({
                                "video_id": vid, "title": snip.get("title", ""),
                                "channelTitle": snip.get("channelTitle", ""),
                                "publishedAt": snip.get("publishedAt", "")
                            })
                        if len(collected_records) >= max_per_keyword: break
                        request = youtube_client.search().list_next(request, response)
                        time.sleep(0.1)
                except Exception as e:
                    st.warning(f"搜尋 '{query}' 錯誤: {e}")
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
            if len(all_video_ids) >= max_total_videos: break
    status_text.empty()
    return list(all_video_ids), video_meta

# =========================
# 3. AI 相關性過濾
# =========================

async def check_relevance_batch_async(movie_title, batch_videos, deepseek_client):
    if not batch_videos: return []
    prompt_items = [f"ID: {v['id']}\nTitle: {v['title']}\nChannel: {v['channel']}" for v in batch_videos]
    prompt_text = "\n---\n".join(prompt_items)
    system_prompt = (
        f"You are a data cleaner. The user is analyzing the movie '{movie_title}'. "
        "Identify which videos are actually discussing this specific movie. "
        "Exclude unrelated videos. Return JSON: {\"vid123\": true, \"vid456\": false}"
    )
    try:
        response = await deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt_text}],
            response_format={"type": "json_object"}, temperature=0.1,
        )
        data = json.loads(response.choices[0].message.content)
        return [vid for vid, is_rel in data.items() if is_rel is True]
    except Exception:
        return [v['id'] for v in batch_videos]

async def filter_videos_by_relevance(movie_title, video_ids, video_meta, deepseek_client):
    to_check = []
    valid_ids = set()
    for vid in video_ids:
        cached = _get_cached_value("relevance_cache", f"{movie_title}_{vid}", CACHE_TTL_RELEVANCE)
        if cached is not None:
            if cached: valid_ids.add(vid)
        else:
            meta = video_meta.get(vid, {})
            to_check.append({"id": vid, "title": meta.get("title", ""), "channel": meta.get("channelTitle", "")})
    
    if to_check:
        batch_size = 20
        tasks = [check_relevance_batch_async(movie_title, to_check[i:i+batch_size], deepseek_client) for i in range(0, len(to_check), batch_size)]
        progress = st.empty()
        progress.info(f"AI 過濾 {len(to_check)} 部影片相關性...")
        results = await asyncio.gather(*tasks)
        for batch_idx, res_list in enumerate(results):
            batch_input = to_check[batch_idx*batch_size : (batch_idx+1)*batch_size]
            rel_set = set(res_list)
            for item in batch_input:
                is_rel = item["id"] in rel_set
                if is_rel: valid_ids.add(item["id"])
                _set_cached_value("relevance_cache", f"{movie_title}_{item['id']}", is_rel)
        progress.empty()
    return list(valid_ids)

# =========================
# 4. 詳情與留言
# =========================

def fetch_video_and_channel_details(video_ids, youtube_client):
    video_extra = {}
    channel_ids = set()
    for i in range(0, len(video_ids), 50):
        try:
            resp = youtube_client.videos().list(part="snippet,contentDetails", id=",".join(video_ids[i:i+50])).execute()
            for item in resp.get("items", []):
                vid = item.get("id")
                snip = item.get("snippet", {})
                ch = snip.get("channelId")
                video_extra[vid] = {
                    "channelId": ch,
                    "defaultLanguage": snip.get("defaultLanguage", ""),
                    "defaultAudioLanguage": snip.get("defaultAudioLanguage", ""),
                    "tags": snip.get("tags", [])
                }
                if ch: channel_ids.add(ch)
        except Exception: pass

    channel_country = {}
    to_fetch = []
    for cid in channel_ids:
        cached = _get_cached_value("channel_cache", cid, CACHE_TTL_CHANNEL)
        if cached: channel_country[cid] = cached
        else: to_fetch.append(cid)
    
    if to_fetch:
        for i in range(0, len(to_fetch), 50):
            try:
                resp = youtube_client.channels().list(part="brandingSettings", id=",".join(to_fetch[i:i+50])).execute()
                for item in resp.get("items", []):
                    cid = item.get("id")
                    country = item.get("brandingSettings", {}).get("channel", {}).get("country")
                    channel_country[cid] = country
                    _set_cached_value("channel_cache", cid, country)
            except Exception: pass
    return video_extra, channel_country

def compute_hk_video_score(video_id, video_meta, video_extra, channel_country_map):
    meta = video_meta.get(video_id, {})
    ext = video_extra.get(video_id, {})
    title = meta.get("title", "")
    tags = " ".join(ext.get("tags", []) or [])
    ch = ext.get("channelId")
    audio = (ext.get("defaultAudioLanguage") or "").lower()
    country = channel_country_map.get(ch)

    score = 0
    if country == "HK": score += 3
    if audio in ("yue", "zh-hk", "zh-yue", "zh-hant-hk"): score += 3
    elif audio.startswith("zh"): score += 1
    
    if any(t in title for t in ["粵語", "廣東話", "粵配", "粵語配音"]): score += 3
    if any(t in title for t in ["香港", "港版", "香港觀眾", "香港反應", "香港首映", "香港上映"]): score += 2
    if ("HK" in title) or ("Hong Kong" in title): score += 1
    if any(t in tags for t in ["粵語", "廣東話", "香港", "HK"]): score += 2
    return score

def get_all_comments(video_ids, youtube_client, max_per_video, video_meta, hk_score_map, video_extra, channel_country_map, max_total_comments):
    all_comments = []
    total_fetched = 0
    progress = st.progress(0, text="抓取留言...")
    
    for i, vid in enumerate(video_ids):
        if total_fetched >= max_total_comments: break
        
        cache_key = f"{vid}_{max_per_video}"
        cached = _get_cached_value("comments_cache", cache_key, CACHE_TTL_COMMENTS)
        raw_recs = []

        if cached is not None:
            raw_recs = cached
        else:
            try:
                req = youtube_client.commentThreads().list(part="snippet", videoId=vid, textFormat="plainText", order="time", maxResults=100)
                fetched_vid = 0
                while req and fetched_vid < max_per_video:
                    if total_fetched + fetched_vid >= max_total_comments: break
                    resp = req.execute()
                    for item in resp.get("items", []):
                        if fetched_vid >= max_per_video: break
                        cmt = item["snippet"]["topLevelComment"]["snippet"]
                        raw_recs.append({
                            "textDisplay": cmt.get("textDisplay", ""),
                            "publishedAt": cmt.get("publishedAt", ""),
                            "likeCount": cmt.get("likeCount", 0)
                        })
                        fetched_vid += 1
                    req = youtube_client.commentThreads().list_next(req, resp)
                    if req and fetched_vid < max_per_video: time.sleep(0.1)
                if raw_recs: _set_cached_value("comments_cache", cache_key, raw_recs)
            except Exception: pass
        
        ch_id = video_extra.get(vid, {}).get("channelId")
        for r in raw_recs:
            all_comments.append({
                "video_id": vid,
                "video_title": video_meta.get(vid, {}).get("title", ""),
                "video_hk_score": hk_score_map.get(vid, 0),
                "video_channel_country": channel_country_map.get(ch_id),
                "comment_text": r["textDisplay"],
                "published_at": r["publishedAt"],
                "like_count": r["likeCount"]
            })
        total_fetched += len(raw_recs)
        progress.progress((i+1)/len(video_ids), text=f"抓取留言... ({min(i+1, len(video_ids))}/{len(video_ids)})")
    progress.empty()
    return pd.DataFrame(all_comments)

# =========================
# 5. DeepSeek 分析
# =========================

async def analyze_comment_deepseek_async(comment_text, deepseek_client, semaphore, max_retries=3):
    if not isinstance(comment_text, str) or len(comment_text.strip()) < 2: # 放寬長度限制
        return {"sentiment": "Invalid", "topic": "N/A", "summary": "Too short."}
    
    # 針對英文或短句的 Prompt 優化
    system_prompt = (
        "You are a professional Hong Kong market sentiment analyst. "
        "Analyze the movie comment. Return JSON with keys: "
        "'sentiment' (Positive/Negative/Neutral), 'topic' (Plot/Acting/Action Design/Visuals/Pace/Overall/N/A), 'summary'. "
        "Treat 'Thank you' or 'Good' as Positive/Overall."
    )
    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await deepseek_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": comment_text}],
                    response_format={"type": "json_object"}, temperature=0.1,
                )
                return json.loads(response.choices[0].message.content)
            except Exception:
                if attempt < max_retries - 1: await asyncio.sleep(2**attempt)
                else: return {"sentiment": "Error", "topic": "Error", "summary": "API Error"}

async def run_all_analyses(df, deepseek_client):
    semaphore = asyncio.Semaphore(50)
    tasks = []
    async def with_index(idx, text):
        res = await analyze_comment_deepseek_async(text, deepseek_client, semaphore)
        return idx, res
    for i, text in enumerate(df["comment_text"]):
        tasks.append(asyncio.create_task(with_index(i, text)))
    
    progress = st.progress(0, text="AI 分析中...")
    results = [None]*len(tasks)
    for done, coro in enumerate(asyncio.as_completed(tasks), 1):
        idx, res = await coro
        results[idx] = res
        progress.progress(done/len(tasks))
    progress.empty()
    return results

# =========================
# 6. 主流程 (核心邏輯修改)
# =========================

def movie_comment_analysis(
    movie_title, start_date, end_date, yt_api_key, deepseek_api_key,
    max_videos_per_keyword=30, max_comments_per_video=50, sample_size=None,
    relax_trad_filter=True, cantonese_threshold=2.0, auto_relax_threshold=True,
    target_min_cantonese=300, prefer_hk_videos=True
):
    target_sample = sample_size if sample_size and sample_size > 0 else 1000
    GLOBAL_MAX_COMMENTS = max(2000, target_sample * 4)
    GLOBAL_MAX_VIDEOS = 150
    SEARCH_KEYWORDS = generate_search_queries(movie_title)

    youtube_client = build("youtube", "v3", developerKey=yt_api_key)
    deepseek_client = openai.AsyncOpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com/v1")

    # 1. 搜尋
    video_ids, video_meta = search_youtube_videos(
        SEARCH_KEYWORDS, youtube_client, max_videos_per_keyword, start_date, end_date,
        add_language_bias=True, region_bias=True, max_total_videos=GLOBAL_MAX_VIDEOS
    )
    if not video_ids: return None, "找不到相關影片。"
    
    # 2. 相關性過濾
    relevant_video_ids = asyncio.run(filter_videos_by_relevance(movie_title, video_ids, video_meta, deepseek_client))
    if not relevant_video_ids: return None, "AI 過濾後無相關影片。"
    
    # 3. 詳情與分數
    video_extra, channel_country_map = fetch_video_and_channel_details(relevant_video_ids, youtube_client)
    hk_score_map = {vid: compute_hk_video_score(vid, video_meta, video_extra, channel_country_map) for vid in relevant_video_ids}
    
    # 排序
    sorted_ids = sorted(relevant_video_ids, key=lambda v: hk_score_map.get(v, 0), reverse=True) if prefer_hk_videos else relevant_video_ids

    # 4. 抓取留言
    df_comments = get_all_comments(
        sorted_ids, youtube_client, max_comments_per_video,
        video_meta, hk_score_map, video_extra, channel_country_map, GLOBAL_MAX_COMMENTS
    )
    if df_comments.empty: return None, "找不到任何留言。"

    st.info(f"已抓取 {len(df_comments)} 則原始留言，開始進行「情境式」篩選...")

    # 5. 語言與情境篩選 (核心修改)
    cc_t2s = OpenCC("t2s")
    cc_s2t = OpenCC("s2t")
    
    # 計算特徵
    df_comments["lang_pred"] = df_comments["comment_text"].apply(lambda x: classify_zh_trad_simp(x, cc_t2s, cc_s2t))
    df_comments["cantonese_score"] = df_comments["comment_text"].apply(score_cantonese)
    
    # 定義篩選邏輯
    def is_target_audience(row):
        text_score = row["cantonese_score"]
        vid_score = row["video_hk_score"]
        lang = row["lang_pred"]
        
        # 條件 A: 文本本身就是強粵語 (無論影片來源)
        if text_score >= cantonese_threshold:
            return True
            
        # 條件 B: 影片是強香港背景 (分數 >= 3)，且留言是繁體中文、英文或未知中文
        # 這能救回 "Thanks for sharing" 或 "謝謝分享"
        if vid_score >= 3 and lang in ["zh-Hant", "zh-unkn", "en"]:
            return True
            
        # 條件 C: 影片是中等香港背景 (分數 >= 1)，且留言是繁體中文 (稍微嚴格一點，不收英文)
        if vid_score >= 1 and lang in ["zh-Hant", "zh-unkn"]:
            # 如果文本分數稍微有一點 (例如有 "系" 或 "d")，也放行
            if text_score >= 0.5:
                return True
                
        return False

    # 初步篩選
    df_comments["is_target"] = df_comments.apply(is_target_audience, axis=1)
    
    # 排除簡體中文 (除非它有很高的粵語分數，例如廣東人打簡體粵語，但這裡我們假設簡體=非目標以保持純淨)
    # 如果想保留廣東省粵語，可移除這行
    df_comments = df_comments[df_comments["lang_pred"] != "zh-Hans"].reset_index(drop=True)
    
    df_filtered = df_comments[df_comments["is_target"]].reset_index(drop=True)

    # 自動放寬邏輯 (現在主要調整的是 text_score 的權重，或者如果樣本太少，我們可以降低 vid_score 的門檻)
    # 這裡簡化為：如果樣本不夠，我們降低對 text_score 的依賴，更多依賴 video_score
    if auto_relax_threshold and len(df_filtered) < target_min_cantonese:
        st.info(f"樣本不足 ({len(df_filtered)})，嘗試放寬條件...")
        # 放寬策略：只要影片有一點香港特徵 (score >= 1) 且是繁體/英文都收
        mask_relaxed = (df_comments["video_hk_score"] >= 1) & (df_comments["lang_pred"].isin(["zh-Hant", "zh-unkn", "en"]))
        df_filtered = df_comments[mask_relaxed].reset_index(drop=True)
        st.info(f"放寬後樣本數：{len(df_filtered)}")

    if df_filtered.empty: return None, "篩選後無符合條件的留言。"

    # 6. 日期與取樣
    df_filtered["published_at"] = pd.to_datetime(df_filtered["published_at"], utc=True, errors="coerce")
    df_filtered["published_at_hk"] = df_filtered["published_at"].dt.tz_convert("Asia/Hong_Kong")
    start_dt = pd.to_datetime(start_date).tz_localize("Asia/Hong_Kong")
    end_dt = pd.to_datetime(end_date).tz_localize("Asia/Hong_Kong") + timedelta(days=1)
    df_filtered = df_filtered[(df_filtered["published_at_hk"] >= start_dt) & (df_filtered["published_at_hk"] < end_dt)].reset_index(drop=True)
    
    if df_filtered.empty: return None, "日期範圍內無留言。"
    
    df_analyze = df_filtered.sample(n=sample_size, random_state=42) if sample_size and 0 < sample_size < len(df_filtered) else df_filtered
    
    # 7. 分析
    analysis_results = asyncio.run(run_all_analyses(df_analyze, deepseek_client))
    final_df = pd.concat([df_analyze.reset_index(drop=True), pd.DataFrame(analysis_results)], axis=1)
    final_df["published_at"] = pd.to_datetime(final_df["published_at"])
    
    return final_df, None

# =========================
# 7. UI
# =========================

st.set_page_config(page_title="YouTube 電影評論 AI 分析（香港粵語優先）", layout="wide")
st.title("🎬 YouTube 電影評論 AI 情感分析（香港粵語優先）")

with st.expander("使用說明"):
    st.markdown("""
    **更新說明：**
    *   已優化篩選邏輯：現在會保留 **香港影片** 底下的 **標準繁體中文** 和 **英文** 留言（例如 "Thanks for sharing" 或 "謝謝分享"）。
    *   已增強粵語識別：支援 "系"、"9"、"d" 等常見網絡用語。
    """)

movie_title = st.text_input("電影名稱", value="九龍城寨之圍城")
col1, col2 = st.columns(2)
with col1: start_date = st.date_input("開始日期", value=datetime.today() - timedelta(days=30))
with col2: end_date = st.date_input("結束日期", value=datetime.today())
yt_api_key = st.text_input("YouTube API Key", type='password')
deepseek_api_key = st.text_input("DeepSeek API Key", type='password')

st.subheader("進階設定")
max_videos = st.slider("每個關鍵字搜尋數", 5, 80, 30)
max_comments = st.slider("每部影片留言數", 10, 200, 80)
sample_size = st.number_input("分析上限", 0, 5000, 500)
cantonese_threshold = st.slider("粵語特徵分數門檻 (針對非香港頻道)", 0.5, 6.0, 2.0)

if st.button("🚀 開始分析"):
    if not all([movie_title, yt_api_key, deepseek_api_key]):
        st.warning("請填寫所有欄位。")
    else:
        with st.spinner("AI 分析中..."):
            df_result, err = movie_comment_analysis(
                movie_title, str(start_date), str(end_date), yt_api_key, deepseek_api_key,
                max_videos, max_comments, sample_size, cantonese_threshold=cantonese_threshold
            )
        if err: st.error(err)
        else:
            st.success("完成！")
            st.dataframe(df_result.head(20), use_container_width=True)
            
            # 簡單圖表展示
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("情感分佈")
                vc = df_result['sentiment'].value_counts()
                st.plotly_chart(px.pie(values=vc.values, names=vc.index, color=vc.index, 
                                     color_discrete_map={'Positive':'#5cb85c','Negative':'#d9534f','Neutral':'#f0ad4e'}), use_container_width=True)
            with c2:
                st.subheader("主題分佈")
                df_topic = df_result[df_result['topic'] != 'N/A']
                if not df_topic.empty:
                    st.plotly_chart(px.bar(df_topic['topic'].value_counts(), orientation='h'), use_container_width=True)

            st.download_button("📥 下載 CSV", df_result.to_csv(index=False, encoding='utf-8-sig'), f"{movie_title}_analysis.csv", "text/csv")
