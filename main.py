import streamlit as st
from googleapiclient.discovery import build
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from soynlp.noun import LCGNounExtractor_v2
from collections import Counter
import re
import pandas as pd
import urllib.request
import os

# --- 1. 한글 폰트 자동 다운로드 (깨짐 방지) ---
@st.cache_data
def download_font():
    font_url = "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Bold.ttf"
    font_path = "NanumGothic-Bold.ttf"
    if not os.path.exists(font_path):
        with st.spinner("한글 폰트를 설정 중입니다..."):
            urllib.request.urlretrieve(font_url, font_path)
    return font_path

FONT_PATH = download_font()

# --- 2. 유튜브 API 기능 ---
def get_video_id(url):
    regex = r'(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})'
    match = re.search(regex, url)
    return match.group(4) if match else None

def get_youtube_comments(api_key, video_id, max_results=100):
    youtube = build('youtube', 'v3', developerKey=api_key)
    comments = []
    likes = []
    
    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(max_results, 100),
            textFormat="plainText"
        )
        response = request.execute()
        
        while response and len(comments) < max_results:
            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                like_count = item['snippet']['topLevelComment']['snippet']['likeCount']
                comments.append(comment)
                likes.append(like_count)
                
            if 'nextPageToken' in response and len(comments) < max_results:
                request = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    pageToken=response['nextPageToken'],
                    maxResults=100,
                    textFormat="plainText"
                )
                response = request.execute()
            else:
                break
                
        return pd.DataFrame({"comment": comments, "likes": likes})
    except Exception as e:
        st.error(f"유튜브 API 호출 실패: {e}")
        return None

# --- 3. 웹 UI 화면 구성 ---
st.set_page_config(page_title="유튜브 댓글 분석기", layout="wide")

st.title("📊 유튜브 댓글 심층 분석 및 워드클라우드")
st.markdown("유튜브 링크와 API 키를 입력하면 인공지능이 한글 명사 키워드를 추출해 시각화합니다.")

st.sidebar.header("🔑 설정")
api_key = st.sidebar.text_input("YouTube API Key", type="password")
video_url = st.sidebar.text_input("유튜브 영상 링크")
max_comments = st.sidebar.slider("수집할 댓글 수", min_value=50, max_value=300, value=100, step=50)

if st.sidebar.button("분석 시작 🚀"):
    if not api_key or not video_url:
        st.warning("API 키와 유튜브 링크를 모두 입력해주세요.")
    else:
        video_id = get_video_id(video_url)
        if not video_id:
            st.error("올바른 유튜브 링크 형식이 아닙니다.")
        else:
            with st.spinner("댓글 데이터를 가져오는 중..."):
                df = get_youtube_comments(api_key, video_id, max_comments)
                
            if df is not None and not df.empty:
                st.success(f"성공적으로 {len(df)}개의 댓글을 수집했습니다!")
                
                # --- 데이터 분석 및 한글 명사 추출 ---
                with st.spinner("텍스트 마이닝 진행 중..."):
                    # 특수문자, 이모지 제거 (한글, 영문, 공백만 유지)
                    cleaned_comments = [re.sub(r'[^가-힣a-zA-Z\s]', '', str(text)) for text in df['comment']]
                    
                    # 순수 파이썬 기반 명사 추출기 가동 (자바 필요 없음)
                    noun_extractor = LCGNounExtractor_v2()
                    try:
                        nouns_chunks = noun_extractor.train_extract(cleaned_comments)
                        all_nouns = [word for word, score in nouns_chunks.items() if len(word) > 1 for _ in range(int(score.frequency))]
                    except:
                        # 댓글 데이터가 적어 형태소 분석 학습이 안 될 경우를 위한 방어용 코드
                        all_nouns = []
                        for text in cleaned_comments:
                            all_nouns.extend([word for word in text.split() if len(word) > 1])
                
                # 대시보드 레이아웃 (좌: 워드클라우드 / 우: 순위 차트)
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("🔠 한글 워드 클라우드")
                    if all_nouns:
                        word_counts = Counter(all_nouns)
                        wc = WordCloud(
                            font_path=FONT_PATH,
                            background_color="white",
                            width=800,
                            height=600,
                            max_words=80
                        ).generate_from_frequencies(word_counts)
                        
                        fig, ax = plt.subplots(figsize=(10, 8))
                        ax.imshow(wc, interpolation='bilinear')
                        ax.axis("off")
                        st.pyplot(fig)
                    else:
                        st.info("댓글에서 추출된 유의미한 키워드가 없습니다.")
                        
                with col2:
                    st.subheader("🔝 주요 키워드 TOP 10")
                    if all_nouns:
                        df_counts = pd.DataFrame(word_counts.most_common(10), columns=['키워드', '빈도수'])
                        st.dataframe(df_counts, use_container_width=True)
                        st.bar_chart(df_counts.set_index('키워드'))
                
                st.markdown("---")
                
                # --- 좋아요 순 베스트 댓글 ---
                st.subheader("🔥 가장 많은 공감을 얻은 댓글 (좋아요 기준)")
                df_sorted = df.sort_values(by="likes", ascending=False).head(5)
                for idx, row in df_sorted.iterrows():
                    st.info(f"👍 **좋아요 {row['likes']}개** \n\n {row['comment']}")
            else:
                st.error("댓글 수집에 실패했습니다. 영상 설정을 확인해 주세요.")
