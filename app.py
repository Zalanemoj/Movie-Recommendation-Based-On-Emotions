import streamlit as st
import pandas as pd
import numpy as np
import pickle
import cv2
import requests
import time
import google.generativeai as genai
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# --- 1. CONFIGURATION & API KEYS ---
# Replace these with your actual keys
TMDB_API_KEY = ""
GEMINI_API_KEY = ""

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.5-flash')

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


# --- 2. ASSET LOADING ---
@st.cache_resource
def load_assets():
    model = load_model('best_model.keras')
    # Load your updated dict (Must have: movie_id, title, genres, popularity, tags)
    movies = pd.DataFrame(pickle.load(open('movie_dict.pkl', 'rb')))
    similarity = pickle.load(open('similarity_matrix.pkl', 'rb'))
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return model, movies, similarity, face_cascade


model, movies, similarity, face_cascade = load_assets()


# --- 3. HELPER FUNCTIONS ---

def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
    try:
        data = requests.get(url).json()
        return "https://image.tmdb.org/t/p/w500/" + data['poster_path']
    except:
        return "https://via.placeholder.com/500x750?text=No+Poster"


def get_gemini_explanation(movie_title, movie_tags, emotion):
    prompt = f"""
    The user is currently feeling {emotion}. I have recommended the movie '{movie_title}'.
    Context (Movie Plot Tags): {movie_tags}.
    In exactly two friendly sentences, explain why this movie matches their mood.
    """
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except:
        return "This movie was selected because its themes align perfectly with your current emotional state."


def detect_and_predict(image_frame):
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 3)
    # Fallback to full image if no face detected
    roi = gray[faces[0][1]:faces[0][1] + faces[0][3], faces[0][0]:faces[0][0] + faces[0][2]] if len(faces) > 0 else gray
    roi = cv2.resize(roi, (48, 48))
    roi = roi.astype('float') / 255.0
    roi = np.expand_dims(img_to_array(roi), axis=0)
    preds = model.predict(roi, verbose=0)
    return EMOTIONS[np.argmax(preds)]


# --- 4. NAVIGATION LOGIC ---
if 'page' not in st.session_state:
    st.session_state.page = 'home'


def go_home(): st.session_state.page = 'home'


# --- 5. MAIN UI ---
st.title("🎬 CinemaAI: Personalized Recommendation Engine")

if st.session_state.page == 'home':
    st.subheader("Select a Service to Begin")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🎭 Recommend by Mood", use_container_width=True):
            st.session_state.page = 'mood'
            st.rerun()
    with col2:
        if st.button("🍿 Recommend by Movie", use_container_width=True):
            st.session_state.page = 'movie'
            st.rerun()
    with col3:
        if st.button("🧠 Emotion Detection Only", use_container_width=True):
            st.session_state.page = 'emotion'
            st.rerun()

# --- 6. PAGE CONTENT ---

if st.session_state.page != 'home':
    if st.button("⬅ Back to Menu"):
        go_home()
        st.rerun()

# A. MOOD-BASED PAGE
if st.session_state.page == 'mood':
    st.header("Step 1: Detect Your Mood")
    source = st.radio("Choose Input:", ["Webcam", "Upload Image"])
    img_file = st.camera_input("Smile!") if source == "Webcam" else st.file_uploader("Upload", type=['jpg', 'png'])

    if img_file:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        mood = detect_and_predict(image)
        st.success(f"Model Detection: **{mood.upper()}**")

        st.header("Step 2: AI Recommendations")
        target_genres = EMOTION_TO_GENRE.get(mood, [])
        mask = movies['genres'].apply(lambda x: any(g in x for g in target_genres))
        recs = movies[mask].sort_values(by='popularity', ascending=False).head(5)

        for row in recs.itertuples():
            c1, c2 = st.columns([1, 3])
            with c1:
                st.image(fetch_poster(row.movie_id))
            with c2:
                st.subheader(row.title)
                with st.spinner("Gemini AI is analyzing plot connection..."):
                    st.write(get_gemini_explanation(row.title, row.tags, mood))
            st.divider()

# B. MOVIE-BASED PAGE
elif st.session_state.page == 'movie':
    st.header("Step 1: Select a Base Movie")
    selected = st.selectbox("Pick a movie:", movies['title'].values)

    if st.button("Step 2: Generate Recommendations"):
        idx = movies[movies['title'] == selected].index[0]
        distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])[1:6]

        st.write("---")
        cols = st.columns(5)
        for i, dist in enumerate(distances):
            with cols[i]:
                m = movies.iloc[dist[0]]
                st.image(fetch_poster(m.movie_id))
                st.caption(m.title)

# C. EMOTION DETECTION PAGE
elif st.session_state.page == 'emotion':
    st.header("Model Validation Mode")
    source = st.radio("Choose Input:", ["Webcam", "Upload Image"])
    img_file = st.camera_input("Analyze") if source == "Webcam" else st.file_uploader("Upload", type=['jpg', 'png'])

    if img_file:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        mood = detect_and_predict(image)
        st.metric(label="CNN Prediction", value=mood.upper())
        st.info("The model analyzes facial features to categorize emotion strictly based on the training set.")