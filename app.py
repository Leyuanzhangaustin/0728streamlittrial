import streamlit as st
import pandas as pd
import numpy as np
import googleapiclient.discovery
from openai import AsyncOpenAI, OpenAIError
import asyncio
from tqdm.asyncio import tqdm as tqdm_asyncio
from datetime import datetime, timedelta
import plotly.express as px
from langdetect import detect, LangDetectException
import json

# --- Constants and Configuration ---

# <<< 新增：繁簡體中文特徵字判斷 >>>
# 建立常用且有代表性的繁簡專用字集合，用於判斷文本類型
# 這些列表不需要非常詳盡，有幾十個常用字就能達到很高的準確率
TC_UNIQUE_CHARS = set("歐體國發見無麼裡蘋語劃讚裡麵")
SC_UNIQUE_CHARS = set("欧体国发见无么里苹语划赞里下面")

# --- Helper Functions ---

def detect_chinese_variant(text: str) -> str:
    """
    使用單一字元特徵判斷文本是繁體、簡體或混合。
    :param text: 輸入的字串。
    :return: "繁體中文", "簡體中文", 或 "混合/未知"。
    """
    if not isinstance(text, str) or not text.strip():
        return "混合/未知"

    tc_count = 0
    sc_count = 0

    for char in text:
        if char in TC_UNIQUE_CHARS:
            tc_count += 1
        elif char in SC_UNIQUE_CHARS:
            sc_count += 1
    
    # 根據命中專用字的數量來判斷
    # 增加一個小小的權重，避免因單一字元誤判
    if tc_count > sc_count:
        return "繁體中文"
    elif sc_count > tc_count:
        return "簡體中文"
    else:
        # 如果數量相等（包括都為0的情況），則視為混合或無法判斷
        return "混合/未知"

def is_chinese(text: str) -> bool:
    """
    使用 langdetect 檢測文本是否為中文。
    """
    try:
        # 只檢測 'zh-cn' 和 'zh-tw'
        lang = detect(text)
        return lang in ['zh-cn', 'zh-tw']
    except LangDetectException:
        # 如果 langdetect 無法識別（例如，純表情符號或太短），則視為非中文
        return False
    except Exception:
        # 處理其他潛在錯誤
        return False

def get_video_comments(youtube, video_id, max_comments):
    """獲取單一影片的留言。"""
    all_comments = []
    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(100, max_comments),
            textFormat="plainText"
        )
        
        while request and len(all_comments) < max_comments:
            response = request.execute()
            for item in response["items"]:
                comment = item["snippet"]["topLevelComment"]["snippet"]
                all_comments.append({
                    "author": comment["authorDisplayName"],
                    "publishedAt": comment["publishedAt"],
                    "textDisplay": comment["textDisplay"]
                })
                if len(all_comments) >= max_comments:
                    break
            
            if 'nextPageToken' in response and len(all_comments) < max_comments:
                request = youtube.commentThreads().list_next(previous_request=request, previous_response=response)
            else:
                break
    except Exception as e:
        st.warning(f"無法獲取影片 ID {video_id} 的留言: {e}", icon="⚠️")
    return all_comments

def search_videos(youtube, query, start_date, end_date, max_videos):
    """根據關鍵字和日期範圍搜索影片。"""
    all_videos = []
    request = youtube.search().list(
        q=query,
        part="snippet",
        type="video",
        maxResults=min(50, max_videos),
        publishedAfter=start_date + "T00:00:00Z",
        publishedBefore=end_date + "T23:59:59Z"
    )

    while request and len(all_videos) < max_videos:
        response = request.execute()
        for item in response['items']:
            all_videos.append({
                'videoId': item['id']['videoId'],
                'title': item['snippet']['title'],
                'publishedAt': item['snippet']['publishedAt']
            })
            if len(all_videos) >= max_videos:
                break
        
        if 'nextPageToken' in response and len(all_videos) < max_videos:
            request = youtube.search().list_next(previous_request=request, previous_response=response)
        else:
            break
            
    return all_videos

# --- AI Analysis Functions ---

# 定義一個 Semaphore，例如，一次最多只允許 10 個並行請求
SEMAPHORE = asyncio.Semaphore(10)

async def analyze_comment_async(comment: str, client: AsyncOpenAI):
    """
    使用 Semaphore 包裹的非同步函式，用於分析單一留言。
    """
    async with SEMAPHORE:
        if not comment or not comment.strip():
            return {"sentiment": "neutral", "positive": 0, "negative": 0, "neutral": 1, "reason": "留言為空"}

        try:
            response = await client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一個情感分析專家。請將以下評論分類為'positive', 'negative', 或 'neutral'。請只用JSON格式回答，包含'sentiment'和'reason'兩個鍵。"},
                    {"role": "user", "content": comment}
                ],
                temperature=0,
                max_tokens=100,
                response_format={"type": "json_object"}
            )
            analysis_text = response.choices[0].message.content
            analysis_json = json.loads(analysis_text)
            
            sentiment = analysis_json.get("sentiment", "neutral").lower()
            
            # 確保情感是三種類型之一
            if sentiment not in ["positive", "negative", "neutral"]:
                sentiment = "neutral"

            return {
                "sentiment": sentiment,
                "positive": 1 if sentiment == "positive" else 0,
                "negative": 1 if sentiment == "negative" else 0,
                "neutral": 1 if sentiment == "neutral" else 0,
                "reason": analysis_json.get("reason", "")
            }
        except OpenAIError as e:
            return {"sentiment": "error", "positive": 0, "negative": 0, "neutral": 0, "reason": f"API 錯誤: {e}"}
        except (json.JSONDecodeError, TypeError):
            return {"sentiment": "error", "positive": 0, "negative": 0, "neutral": 0, "reason": "無法解析AI回應"}
        except Exception as e:
            return {"sentiment": "error", "positive": 0, "negative": 0, "neutral": 0, "reason": f"未知錯誤: {e}"}

async def run_all_analyses(df: pd.DataFrame, client: AsyncOpenAI):
    """使用 asyncio.gather 和 Semaphore 執行所有留言的情感分析。"""
    tasks = [analyze_comment_async(comment, client) for comment in df['textDisplay']]
    
    analysis_results = await tqdm_asyncio.gather(
        *tasks, 
        desc="AI Sentiment Analysis (Concurrent)"
    )
    
    return analysis_results

# --- Visualization Functions ---

def create_sunburst_chart(df: pd.DataFrame):
    """
    創建一個 Plotly 旭日圖，顯示情感和語言變體的層級分佈。
    """
    # 確保情感和語言變體欄位存在
    if 'sentiment' not in df.columns or 'script_variant' not in df.columns:
        st.warning("缺少 'sentiment' 或 'script_variant' 欄位，無法生成旭日圖。")
        return None

    # 處理 'error' 情感，將其歸類為 'neutral' 以便於視覺化
    df_plot = df.copy()
    df_plot['sentiment'] = df_plot['sentiment'].replace('error', 'neutral')
    
    # 創建旭日圖
    fig = px.sunburst(
        df_plot,
        path=['sentiment', 'script_variant'], # <<< 修改：增加層級
        title="情感與語言變體分佈旭日圖",
        color='sentiment',
        color_discrete_map={
            'positive': '#2ca02c', # 綠色
            'negative': '#d62728', # 紅色
            'neutral': '#7f7f7f'   # 灰色
        }
    )
    fig.update_traces(textinfo="label+percent parent")
    fig.update_layout(margin=dict(t=50, l=10, r=10, b=10))
    return fig

# --- Main Application Logic ---

@st.cache_data(ttl=3600)
def movie_comment_analysis(movie_title, start_date, end_date, yt_api_key, deepseek_api_key, max_videos, max_comments, sample_size):
    try:
        youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=yt_api_key)
        
        # 1. 搜索影片
        videos = search_videos(youtube, f"{movie_title} 預告", start_date, end_date, max_videos)
        if not videos:
            return None, "在指定日期範圍內找不到相關影片。"

        # 2. 獲取所有影片的留言
        all_comments = []
        for video in videos:
            comments = get_video_comments(youtube, video['videoId'], max_comments)
            all_comments.extend(comments)
        
        if not all_comments:
            return None, "找到了影片，但無法獲取任何留言。"

        df_comments = pd.DataFrame(all_comments)
        df_comments.drop_duplicates(subset=['textDisplay'], inplace=True)

        # 3. 語言檢測與篩選
        st.write(f"初步獲取 {len(df_comments)} 則不重複留言，開始進行語言篩選...")
        df_comments['is_chinese'] = df_comments['textDisplay'].apply(is_chinese)
        df_chinese = df_comments[df_comments['is_chinese']].copy()
        
        if df_chinese.empty:
            return None, "過濾後沒有找到任何中文留言。"
        
        # <<< 新增：應用繁簡體判斷函式 >>>
        df_chinese['script_variant'] = df_chinese['textDisplay'].apply(detect_chinese_variant)

        # 4. 留言抽樣
        num_to_analyze = min(len(df_chinese), sample_size)
        df_analyze = df_chinese.sample(n=num_to_analyze, random_state=42)

        # 5. AI 情感分析
        if not deepseek_api_key:
            return None, "請在左側輸入 DeepSeek API 金鑰以進行情感分析。"
            
        deepseek_client = AsyncOpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com/v1")
        
        analysis_results = asyncio.run(run_all_analyses(df_analyze, deepseek_client))
        
        df_analysis_results = pd.DataFrame(analysis_results)
        
        # 6. 合併結果
        df_result = df_analyze.reset_index(drop=True).join(df_analysis_results)
        
        return df_result, None

    except Exception as e:
        return None, f"發生嚴重錯誤: {e}"

# --- Streamlit UI ---

st.set_page_config(page_title="電影社群口碑分析器", layout="wide")
st.title("🎬 電影社群口碑分析器")
st.markdown("輸入電影名稱，本工具將自動從 YouTube 抓取相關預告片的留言，並使用 AI 進行情感分析，幫助您快速了解大眾口碑。")

# Sidebar for inputs
with st.sidebar:
    st.header("⚙️ 分析設定")
    
    yt_api_key = st.text_input("Google (YouTube) API Key", type="password", help="請輸入您的 YouTube Data API v3 金鑰。")
    deepseek_api_key = st.text_input("DeepSeek API Key", type="password", help="請輸入您的 DeepSeek API 金鑰。")

    movie_title = st.text_input("電影名稱", "沙丘")
    
    today = datetime.now()
    one_month_ago = today - timedelta(days=30)
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("開始日期", one_month_ago)
    with col2:
        end_date = st.date_input("結束日期", today)

    st.subheader("資料量控制")
    max_videos = st.slider("最大影片搜索數量", 1, 50, 5, help="從 YouTube 搜索多少部相關影片來抓取留言。")
    max_comments = st.slider("每部影片最大留言數", 50, 500, 100, help="從每部影片中最多抓取多少則留言。")
    sample_size = st.slider("AI 分析樣本數", 50, 1000, 200, help="從所有中文留言中隨機抽取多少則進行 AI 情感分析。")

    analyze_button = st.button("🚀 開始分析", use_container_width=True, type="primary")

if analyze_button:
    if not yt_api_key or not deepseek_api_key:
        st.error("請務必在左側設定中輸入 YouTube 和 DeepSeek 的 API 金鑰！")
    else:
        with st.spinner("正在執行分析，請稍候... (可能需要幾分鐘)"):
            df_result, err = movie_comment_analysis(
                movie_title, str(start_date), str(end_date),
                yt_api_key, deepseek_api_key,
                max_videos, max_comments, sample_size
            )

        if err:
            st.error(f"分析失敗：{err}")
        elif df_result is not None:
            st.success("分析完成！")
            
            # --- 結果展示 ---
            total_comments = len(df_result)
            positive_count = df_result['positive'].sum()
            negative_count = df_result['negative'].sum()
            neutral_count = df_result['neutral'].sum()
            error_count = (df_result['sentiment'] == 'error').sum()

            st.header("📊 整體情感分佈")
            
            # 指標卡
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("正面留言 👍", f"{positive_count}", f"{positive_count/total_comments:.1%}")
            col2.metric("負面留言 👎", f"{negative_count}", f"-{negative_count/total_comments:.1%}")
            col3.metric("中性留言 😐", f"{neutral_count}", f"{neutral_count/total_comments:.1%}")
            col4.metric("分析失敗數 ❌", f"{error_count}", "應為 0")

            # 視覺化圖表
            sunburst_fig = create_sunburst_chart(df_result)
            if sunburst_fig:
                st.plotly_chart(sunburst_fig, use_container_width=True)

            # --- 留言詳情與篩選 ---
            st.header("📜 留言詳情")

            # <<< 新增：語言變體篩選器 >>>
            variant_options = ["全部"] + df_result['script_variant'].unique().tolist()
            selected_variant = st.selectbox("篩選語言變體:", options=variant_options)

            # 根據選擇進行篩選
            if selected_variant == "全部":
                df_display = df_result
            else:
                df_display = df_result[df_result['script_variant'] == selected_variant]

            # 顯示篩選後的資料
            st.dataframe(df_display[[
                'sentiment', 
                'script_variant', # <<< 新增：顯示語言變體欄
                'textDisplay', 
                'reason', 
                'author', 
                'publishedAt'
            ]], use_container_width=True)
            
            st.info(f"共顯示 {len(df_display)} / {total_comments} 則分析後的留言。")
