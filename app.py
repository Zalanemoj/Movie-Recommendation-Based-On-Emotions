import streamlit as st
import pandas as pd
import numpy as np
import pickle
import cv2
import requests
import google.generativeai as genai
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# --- 1. CONFIG & API KEYS ---
# Replace with your actual keys
TMDB_API_KEY = "6e6405f20790c821fd18d3fb0ab9df27"
GEMINI_API_KEY = "AIzaSyAD4BGngvhSRZhJHAJ_Cz3ue4Cb9RHMOIo"

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

# --- 2. ASSET LOADING (Updated Paths) ---
@st.cache_resource
def load_assets():
    # Looking inside 'Model-Files' folder based on your new structure
    model = load_model('Model-Files/best_model.keras')
    movies = pd.DataFrame(pickle.load(open('Model-Files/movie_dict.pkl', 'rb')))
    similarity = pickle.load(open('Model-Files/similarity_matrix.pkl', 'rb'))
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
    prompt = f"Feeling {emotion}. Movie: {movie_title}. Plot: {movie_tags}. Explain why it matches in 2 sentences."
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except:
        return "Matches your mood!"

def detect_and_predict(image_frame):
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 3)
    roi = gray[faces[0][1]:faces[0][1]+faces[0][3], faces[0][0]:faces[0][0]+faces[0][2]] if len(faces) > 0 else gray
    roi = cv2.resize(roi, (48, 48))
    roi = roi.astype('float') / 255.0
    roi = np.expand_dims(img_to_array(roi), axis=0)
    preds = model.predict(roi, verbose=0)
    return EMOTIONS[np.argmax(preds)]

# --- 4. STREAMLIT UI ---
if 'page' not in st.session_state: st.session_state.page = 'home'

st.title("🎬 CinemaAI: Personalized Recommender")

if st.session_state.page == 'home':
    col1, col2, col3 = st.columns(3)
    if col1.button("🎭 Recommend by Mood"): st.session_state.page = 'mood'; st.rerun()
    if col2.button("🍿 Recommend by Movie"): st.session_state.page = 'movie'; st.rerun()
    if col3.button("🧠 Detect Emotion"): st.session_state.page = 'emotion'; st.rerun()

if st.session_state.page != 'home':
    if st.button("⬅ Back"): st.session_state.page = 'home'; st.rerun()

if st.session_state.page == 'mood':
    img_file = st.camera_input("Capture Mood")
    if img_file:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        mood = detect_and_predict(image)
        st.success(f"Mood: {mood.upper()}")
        
        target_genres = EMOTION_TO_GENRE.get(mood, [])
        mask = movies['genres'].apply(lambda x: any(g in x for g in target_genres))
        recs = movies[mask].sort_values(by='popularity', ascending=False).head(3)
        
        for row in recs.itertuples():
            c1, c2 = st.columns([1, 3])
            c1.image(fetch_poster(row.movie_id))
            with c2:
                st.subheader(row.title)
                st.write(get_gemini_explanation(row.title, row.tags, mood))
            st.divider()

elif st.session_state.page == 'movie':
    selected = st.selectbox("Pick a movie:", movies['title'].values)
    if st.button("Recommend"):
        idx = movies[movies['title'] == selected].index[0]
        distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])[1:6]
        cols = st.columns(5)
        for i, dist in enumerate(distances):
            with cols[i]:
                m = movies.iloc[dist[0]]
                st.image(fetch_poster(m.movie_id))
                st.caption(m.title)
