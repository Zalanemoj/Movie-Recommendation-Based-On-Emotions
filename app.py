import streamlit as st
import pandas as pd
import numpy as np
import pickle
import cv2
import requests
import os
import kagglehub
from groq import Groq
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

TMDB_API_KEY = st.secrets["TMDB_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

os.environ['KAGGLE_USERNAME'] = st.secrets["KAGGLE_USERNAME"]
os.environ['KAGGLE_KEY'] = st.secrets["KAGGLE_KEY"]

client = Groq(api_key=GROQ_API_KEY)

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EMOTION_TO_GENRE = {
    'happy': ['Animation', 'Adventure', 'Comedy'],
    'sad': ['Drama', 'Romance', 'Musical'],
    'angry': ['Action', 'Thriller', 'Crime'],
    'fear': ['Horror', 'Mystery'],
    'surprise': ['ScienceFiction', 'Fantasy', 'Mystery'],
    'neutral': ['Documentary', 'Biography', 'History', 'Drama'],
    'disgust': ['Family', 'Animation']
}

@st.cache_resource
def load_assets():
    model_handle = "krishnapalsinhzala13/emotion-detector/keras/default"
    with st.spinner("Downloading Model from Kaggle..."):
        model_dir = kagglehub.model_download(model_handle)
    
    model = load_model(os.path.join(model_dir, 'best_model (1).keras'))
    movies = pd.DataFrame(pickle.load(open('Model-Files/movie_dict.pkl', 'rb')))
    similarity = pickle.load(open('Model-Files/similarity_matrix.pkl', 'rb'))
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    return model, movies, similarity, face_cascade

model, movies, similarity, face_cascade = load_assets()

def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
    try:
        data = requests.get(url).json()
        return "https://image.tmdb.org/t/p/w500/" + data['poster_path']
    except:
        return "https://via.placeholder.com/500x750?text=No+Poster"

def get_groq_explanation(movie_title, movie_tags, emotion):
    prompt = (f"User is feeling {emotion}. Explain in 2 sentences why '{movie_title}' "
              f"is a perfect match based on these keywords: {movie_tags}.")
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7, max_tokens=150
        )
        return completion.choices[0].message.content
    except:
        return "This movie matches your mood perfectly!"

def get_mood_tips(emotion):
    prompt = f"A user is feeling {emotion}. Provide 3 short, practical tips to improve or manage this mood effectively."
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=1, max_tokens=250
        )
        return completion.choices[0].message.content
    except:
        return "Take a deep breath and stay positive!"

def detect_and_predict(image_frame):
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 3)
    roi = gray[faces[0][1]:faces[0][1]+faces[0][3], faces[0][0]:faces[0][0]+faces[0][2]] if len(faces) > 0 else gray
    roi = cv2.resize(roi, (48, 48))
    roi = roi.astype('float') / 255.0
    roi = np.expand_dims(img_to_array(roi), axis=0)
    preds = model.predict(roi, verbose=0)[0]
    emotion = EMOTIONS[np.argmax(preds)]
    emotion_confidence = float(np.max(preds)) * 100  # confidence % of detected emotion
    all_scores = {EMOTIONS[i]: round(float(preds[i]) * 100, 1) for i in range(len(EMOTIONS))}
    return emotion, emotion_confidence, all_scores

def get_genre_match_confidence(movie_genres, target_genres):
    """How well does a movie's genres match the expected emotion genres (0-100%)"""
    if not target_genres or not movie_genres:
        return 0.0
    matched = sum(1 for g in target_genres if g in movie_genres)
    return round((matched / len(target_genres)) * 100, 1)

def confidence_badge(score):
    """Return colored label based on confidence score"""
    if score >= 70:
        return "🟢 High Match"
    elif score >= 40:
        return "🟡 Medium Match"
    else:
        return "🔴 Low Match"

if 'page' not in st.session_state: st.session_state.page = 'home'

if 'movie_offset' not in st.session_state: st.session_state.movie_offset = 0
if 'show_movie_recs' not in st.session_state: st.session_state.show_movie_recs = False
if 'last_selected_movie' not in st.session_state: st.session_state.last_selected_movie = ""
    
if 'mood_offset' not in st.session_state: st.session_state.mood_offset = 0
if 'last_img_bytes' not in st.session_state: st.session_state.last_img_bytes = None

st.set_page_config(page_title="CinemaAI", layout="wide")
st.title("🎬 CinemaAI: Personalized Recommender")


if st.session_state.page == 'home':
    st.subheader("Welcome to your Personalized Cinema Experience!")
    st.markdown("""
    ### How to use this App:
    1. ** Mood-Based Movie**: We scan your face to detect emotion and suggest movies. **Click 'Show Next' to see more options.**
    2. ** Only Movie Recommender**: Select a movie you love. We'll show similar ones. **Pagination is now fixed!**
    3. ** Only Emotion Detection**: Detect your mood and get AI wellness tips.
    """)
    col1, col2, col3 = st.columns(3)
    if col1.button(" Mood-Based Movie", use_container_width=True): st.session_state.page = 'mood_movie'; st.rerun()
    if col2.button(" Only Movie Recommender", use_container_width=True): st.session_state.page = 'only_movie'; st.rerun()
    if col3.button(" Only Emotion Detection", use_container_width=True): st.session_state.page = 'only_emotion'; st.rerun()

else:
    if st.sidebar.button("⬅ Back to Menu"):
        st.session_state.show_movie_recs = False
        st.session_state.movie_offset = 0
        st.session_state.mood_offset = 0
        st.session_state.last_img_bytes = None
        st.session_state.page = 'home'
        st.rerun()

    if st.session_state.page == 'mood_movie':
        st.header("Movie Recommendations Based on Your Emotion")
        
        tab1, tab2 = st.tabs([" Use Camera", " Upload Image"])
        with tab1: cam_file = st.camera_input("Capture face")
        with tab2: up_file = st.file_uploader("Upload image", type=['jpg', 'png', 'jpeg'])
        
        img_file = cam_file if cam_file else up_file

        if img_file:
            file_bytes_obj = img_file.getvalue()
            
            if st.session_state.last_img_bytes != file_bytes_obj:
                st.session_state.last_img_bytes = file_bytes_obj
                st.session_state.mood_offset = 0

            file_bytes = np.asarray(bytearray(file_bytes_obj), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            mood, emotion_conf, all_scores = detect_and_predict(image)

            # --- Emotion Confidence Display ---
            st.success(f"Detected Mood: **{mood.upper()}**")

            col_conf, col_bar = st.columns([1, 2])
            with col_conf:
                st.metric(label="🎭 Emotion Confidence", value=f"{emotion_conf:.1f}%")
            with col_bar:
                st.markdown("**All Emotion Scores:**")
                for emo, score in sorted(all_scores.items(), key=lambda x: -x[1]):
                    st.progress(int(score), text=f"{emo.capitalize()}: {score}%")

            st.divider()

            target_genres = EMOTION_TO_GENRE.get(mood, [])
            mask = movies['genres'].apply(lambda x: any(g in x for g in target_genres))
            all_recs = movies[mask].sort_values(by='popularity', ascending=False)
            
            start = st.session_state.mood_offset
            end = start + 5
            batch_recs = all_recs.iloc[start:end]
            
            if batch_recs.empty:
                st.warning("No more movies to show for this mood!")
            else:
                st.subheader(f"Movies for a {mood} day (Showing {start+1}-{end}):")
                for row in batch_recs.itertuples():
                    c1, c2 = st.columns([1, 4])
                    with c1:
                        st.image(fetch_poster(row.movie_id))
                    with c2:
                        st.subheader(row.title)

                        # --- Movie-Emotion Match Confidence ---
                        match_conf = get_genre_match_confidence(row.genres, target_genres)
                        badge = confidence_badge(match_conf)
                        st.markdown(f"**Emotion Match:** {badge} `{match_conf}%`")
                        st.progress(int(match_conf), text=f"Genre overlap with '{mood}' mood")

                        with st.spinner("AI is thinking..."):
                            st.write(get_groq_explanation(row.title, row.tags, mood))
                    st.divider()

                if st.button("Already seen these? Show Next 5 ➡️", key="next_mood_btn"):
                    st.session_state.mood_offset += 5
                    st.rerun()

    elif st.session_state.page == 'only_movie':
        st.header("Similar Movie Recommender")
        
        selected = st.selectbox("Pick a movie you liked:", movies['title'].values)
        
        if selected != st.session_state.last_selected_movie:
            st.session_state.show_movie_recs = False
            st.session_state.movie_offset = 0
            st.session_state.last_selected_movie = selected

        if st.button("Get Recommendations"):
            st.session_state.show_movie_recs = True
            st.session_state.movie_offset = 0 

        if st.session_state.show_movie_recs:
            idx = movies[movies['title'] == selected].index[0]
            distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])

            start = st.session_state.movie_offset

            slice_start = 1 + start
            slice_end = slice_start + 5
            
            current_batch = distances[slice_start:slice_end]
            
            cols = st.columns(5)
            for i, dist in enumerate(current_batch):
                with cols[i]:
                    m = movies.iloc[dist[0]]
                    similarity_score = round(dist[1] * 100, 1)
                    badge = confidence_badge(similarity_score)
                    st.image(fetch_poster(m.movie_id))
                    st.caption(m.title)
                    st.progress(int(similarity_score), text=f"{similarity_score}%")
                    st.markdown(f"<center>{badge}</center>", unsafe_allow_html=True)
            
            st.divider()
            
            if st.button("Already seen these? Show Next 5 ➡️", key="next_movie_btn"):
                st.session_state.movie_offset += 5
                st.rerun()
    elif st.session_state.page == 'only_emotion':
        st.header("Emotion Detection & AI Tips")
        
        tab1, tab2 = st.tabs([" Use Camera", " Upload Image"])
        with tab1: cam_file = st.camera_input("Capture face for analysis")
        with tab2: up_file = st.file_uploader("Upload face image", type=['jpg', 'png', 'jpeg'])
        
        img_file = cam_file if cam_file else up_file

        if img_file:
            file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            mood, emotion_conf, all_scores = detect_and_predict(image)
            
            c1, c2 = st.columns([1, 2])
            with c1: st.image(image, channels="BGR", caption="Input Image", width=300)
            with c2:
                st.metric(label="Predicted Mood", value=mood.upper(), delta=f"{emotion_conf:.1f}% confident")
                st.markdown("**All Emotion Probabilities:**")
                for emo, score in sorted(all_scores.items(), key=lambda x: -x[1]):
                    st.progress(int(score), text=f"{emo.capitalize()}: {score}%")
                st.markdown("### AI Wellness Tips")
                with st.spinner("Generating tips..."):
                    st.info(get_mood_tips(mood))

