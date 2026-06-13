import streamlit as st
from googleapiclient.discovery import build
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from konlpy.tag import Okt
from collections import Counter
import re
import pandas as pd
import urllib.request
import os

# --- 1. 한글 폰트 다운로드 설정 (스트림릿 클라우드용) ---
@st.cache_data
def download_font():
    font_url = "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Bold.ttf"
    font_path = "NanumGothic-Bold.ttf"
    if not os.path.exists(font_path):
        with st.spinner("한글 폰트를 다운로드 중입니다..."):
            urllib.request.urlretrieve(font_url, font_path)
    return font_path

FONT_PATH = download_font()

# --- 2. 유튜브 API 댓글 수집 함수 ---
def get_video_id(url):
    """유튜브 URL에서 비디오 ID 추출"""
    regex = r'(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})'
    match = re.search(regex, url)
    if match:
        return match.group(4)
    return None

def get_youtube_comments(api_key, video_id, max_results=100):
    """유튜브 댓글 및 좋아요 수 수집"""
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
                
            # 다음 페이지가 있고 목표 수치에 도달하지 못했을 때 반복
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
        st.error(f"API 호출 중 오류가 발생했습니다: {e}")
        return None

# --- 3. 스트림릿 UI 구성 ---
st.set_page_config(page_title="유튜브 댓글 심층 분석기", layout="wide")

st.title("📊 유튜브 댓글 심층 분석 및 워드클라우드")
st.markdown("유튜브 링크와 API 키를 입력해 댓글의 핵심 키워드와 반응을 분석해보세요.")

# 사이드바 설정
st.sidebar.header("🔑 설정 및 입력")
api_key = st.sidebar.text_input("YouTube API Key", type="password", help="Google Cloud Console에서 발급받은 API 키를 입력하세요.")
video_url = st.sidebar.text_input("유튜브 영상 링크", placeholder="https://www.youtube.com/watch?v=...")
max_comments = st.sidebar.slider("수집할 댓글 수", min_value=50, max_value=500, value=100, step=50)

if st.sidebar.button("분석 시작🚀"):
    if not api_key:
        st.warning("API 키를 입력해주세요.")
    elif not video_url:
        st.warning("유튜브 영상 주소를 입력해주세요.")
    else:
        video_id = get_video_id(video_url)
        if not video_id:
            st.error("올바른 유튜브 URL 형식이 아닙니다.")
        else:
            with st.spinner("유튜브에서 댓글을 긁어오는 중입니다..."):
                df = get_youtube_comments(api_key, video_id, max_comments)
                
            if df is not None and not df.empty:
                st.success(f"성공적으로 {len(df)}개의 댓글을 수집했습니다!")
                
                # --- 데이터 전처리 및 형태소 분석 ---
                with st.spinner("한글 키워드 분석 중..."):
                    okt = Okt()
                    all_nouns = []
                    
                    for text in df['comment']:
                        # 한글, 영문, 공백만 남기기
                        cleaned_text = re.sub(r'[^가-힣a-zA-Z\s]', '', str(text))
                        # 명사 추출 (단어 길이가 2자 이상인 것만)
                        nouns = [word for word in okt.nouns(cleaned_text) if len(word) > 1]
                        all_nouns.extend(nouns)
                
                # 대시보드 레이아웃 구성 (2단 배치)
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("🔠 한글 워드 클라우드")
                    if all_nouns:
                        word_counts = Counter(all_nouns)
                        
                        # 워드클라우드 생성
                        wc = WordCloud(
                            font_path=FONT_PATH,
                            background_color="white",
                            width=800,
                            height=600,
                            max_words=100
                        ).generate_from_frequencies(word_counts)
                        
                        # 시각화 출력
                        fig, ax = plt.subplots(figsize=(10, 8))
                        ax.imshow(wc, interpolation='bilinear')
                        ax.axis("off")
                        st.pyplot(fig)
                    else:
                        st.info("추출된 유의미한 한글 명사가 없습니다.")
                        
                with col2:
                    st.subheader("🔝 가장 많이 등장한 키워드 TOP 10")
                    if all_nouns:
                        df_counts = pd.DataFrame(word_counts.most_common(10), columns=['키워드', '빈도수'])
                        st.dataframe(df_counts, use_container_width=True)
                        
                        # 간단한 바 차트
                        st.bar_chart(df_counts.set_index('키워드'))
                
                st.markdown("---")
                
                # --- 좋아요 순 탑 댓글 분석 ---
                st.subheader("🔥 가장 공감을 많이 받은 댓글 (좋아요 순)")
                df_sorted = df.sort_values(by="likes", ascending=False).head(5)
                for idx, row in df_sorted.iterrows():
                    st.info(f"👍 **좋아요 {row['likes']}개** \n\n {row['comment']}")
                    
            else:
                st.error("댓글을 가져오지 못했거나 댓글이 없는 영상입니다.")
