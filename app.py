import streamlit as st
import pandas as pd
import numpy as np
import pickle
import cv2
import requests
from groq import Groq
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

TMDB_API_KEY = ""
GROQ_API_KEY = "" 

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
    model = load_model('Model-Files/best_model.keras')
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
    """Generates the 'points' or explanations using Groq AI."""
    prompt = (
        f"A user is feeling {emotion}. I recommended the movie '{movie_title}' "
        f"based on these plot tags: {movie_tags}. "
        f"In exactly 2 sentences, explain why this movie is a perfect match."
    )
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"This {emotion} movie is a great match for your current mood!"

def detect_and_predict(image_frame):
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 3)
    roi = gray[faces[0][1]:faces[0][1]+faces[0][3], faces[0][0]:faces[0][0]+faces[0][2]] if len(faces) > 0 else gray
    roi = cv2.resize(roi, (48, 48))
    roi = roi.astype('float') / 255.0
    roi = np.expand_dims(img_to_array(roi), axis=0)
    preds = model.predict(roi, verbose=0)
    return EMOTIONS[np.argmax(preds)]

if 'page' not in st.session_state: st.session_state.page = 'home'

st.title("🎬 CinemaAI: Personalized Recommender")

if st.session_state.page == 'home':
    st.subheader("How would you like to find a movie today?")
    col1, col2, col3 = st.columns(3)
    if col1.button("🎭 Mood-Based", use_container_width=True): st.session_state.page = 'mood'; st.rerun()
    if col2.button("🍿 Similar-Movie", use_container_width=True): st.session_state.page = 'movie'; st.rerun()
    if col3.button("🧠 Test Emotion", use_container_width=True): st.session_state.page = 'emotion'; st.rerun()

else:
    if st.button("⬅ Back to Menu"): st.session_state.page = 'home'; st.rerun()

    if st.session_state.page == 'mood':
        img_file = st.camera_input("Capture your face to detect mood")
        if img_file:
            file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            mood = detect_and_predict(image)
            st.success(f"Detected Mood: **{mood.upper()}**")
            
            target_genres = EMOTION_TO_GENRE.get(mood, [])
            mask = movies['genres'].apply(lambda x: any(g in x for g in target_genres))
            recs = movies[mask].sort_values(by='popularity', ascending=False).head(3)
            
            for row in recs.itertuples():
                c1, c2 = st.columns([1, 3])
                c1.image(fetch_poster(row.movie_id))
                with c2:
                    st.subheader(row.title)
                    with st.spinner("Analyzing match..."):
                        st.write(get_groq_explanation(row.title, row.tags, mood))
                st.divider()

    elif st.session_state.page == 'movie':
        selected = st.selectbox("Pick a movie you liked:", movies['title'].values)
        if st.button("Get Recommendations"):
            idx = movies[movies['title'] == selected].index[0]
            distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])[1:6]
            cols = st.columns(5)
            for i, dist in enumerate(distances):
                with cols[i]:
                    m = movies.iloc[dist[0]]
                    st.image(fetch_poster(m.movie_id))
                    st.caption(m.title)

    elif st.session_state.page == 'emotion':
        img_file = st.camera_input("Face the camera")
        if img_file:
            file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            mood = detect_and_predict(image)
            st.metric(label="CNN Model Prediction", value=mood.upper())

