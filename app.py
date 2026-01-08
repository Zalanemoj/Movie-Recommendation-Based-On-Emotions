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

# --- STEP 1: AUTHENTICATION & SECRETS ---
TMDB_API_KEY = st.secrets["TMDB_API_KEY"] 
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
os.environ['KAGGLE_USERNAME'] = st.secrets["KAGGLE_USERNAME"]
os.environ['KAGGLE_KEY'] = st.secrets["KAGGLE_KEY"]

client = Groq(api_key=GROQ_API_KEY)

# Configuration for Mood-to-Genre mapping
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

# --- STEP 2: LOADING ASSETS (KAGGLE MODEL + GITHUB DATA) ---
@st.cache_resource
def load_assets():
    model_handle = "krishnapalsinhzala13/emotion-detector/keras/default"
    with st.spinner("Downloading model from Kaggle Registry..."):
        model_dir = kagglehub.model_download(model_handle)
    
    # Load model from Kaggle
    model = load_model(os.path.join(model_dir, 'best_model (1).keras'))
    
    # Load data from local GitHub repo
    movies = pd.DataFrame(pickle.load(open('Model-Files/movie_dict.pkl', 'rb')))
    similarity = pickle.load(open('Model-Files/similarity_matrix.pkl', 'rb'))
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return model, movies, similarity, face_cascade

model, movies, similarity, face_cascade = load_assets()

# --- STEP 3: HELPER FUNCTIONS ---
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
    try:
        data = requests.get(url).json()
        return "https://image.tmdb.org/t/p/w500/" + data['poster_path']
    except:
        return "https://via.placeholder.com/500x750?text=No+Poster"

def get_groq_explanation(movie_title, movie_tags, emotion):
    prompt = (f"User is {emotion}. Explain in 2 sentences why '{movie_title}' "
              f"is perfect based on: {movie_tags}.")
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7, max_tokens=150
        )
        return completion.choices[0].message.content
    except:
        return f"This movie is a great match for your current mood!"

def get_mood_tips(emotion):
    """Generates AI tips to help change or manage the detected mood."""
    prompt = f"A user is feeling {emotion}. Give 3 short, practical tips (1 sentence each) to help them improve or manage their mood effectively."
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7, max_tokens=200
        )
        return completion.choices[0].message.content
    except:
        return "Take a deep breath and listen to some upbeat music!"

def detect_and_predict(image_frame):
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 3)
    roi = gray[faces[0][1]:faces[0][1]+faces[0][3], faces[0][0]:faces[0][0]+faces[0][2]] if len(faces) > 0 else gray
    roi = cv2.resize(roi, (48, 48))
    roi = roi.astype('float') / 255.0
    roi = np.expand_dims(img_to_array(roi), axis=0)
    preds = model.predict(roi, verbose=0)
    return EMOTIONS[np.argmax(preds)]

# --- STEP 4: STATE MANAGEMENT ---
if 'page' not in st.session_state: st.session_state.page = 'home'
if 'recommend_offset' not in st.session_state: st.session_state.recommend_offset = 1

st.set_page_config(page_title="CinemaAI", layout="wide")
st.title("🎬 CinemaAI: Personalized Recommender")

# --- STEP 5: NAVIGATION & LOGIC ---
if st.session_state.page == 'home':
    st.subheader("Welcome to CinemaAI!")
    
    # Usage Instructions
    st.markdown("""
    ### How to use this app:
    1. **🎭 Mood-Based**: Upload a photo or capture your face. Our AI will detect your emotion and recommend movies to match or improve your mood.
    2. **🍿 Similar-Movie**: Select a movie you already love. We will suggest similar titles based on plot, tags, and genres.
    3. **✨ Explore More**: If you've seen our first set of suggestions, use the 'Suggest Next' button to discover more!
    """)
    
    col1, col2 = st.columns(2)
    if col1.button("🎭 Mood-Based", use_container_width=True):
        st.session_state.page = 'mood'; st.rerun()
    if col2.button("🍿 Similar-Movie", use_container_width=True):
        st.session_state.page = 'movie'; st.session_state.recommend_offset = 1; st.rerun()

else:
    if st.sidebar.button("⬅ Back to Menu"):
        st.session_state.page = 'home'; st.rerun()

    # --- MOOD RECOGNITION SECTION ---
    if st.session_state.page == 'mood':
        img_file = st.file_uploader("Upload or Capture face", type=['jpg', 'png'])
        if img_file:
            file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            mood = detect_and_predict(image)
            st.success(f"Detected Mood: {mood.upper()}")
            
            # AI Mood Tips
            with st.expander("💡 AI Tips to improve your mood"):
                st.write(get_mood_tips(mood))

            st.write("### Recommended Movies for your mood:")
            target_genres = EMOTION_TO_GENRE.get(mood, [])
            mask = movies['genres'].apply(lambda x: any(g in x for g in target_genres))
            recs = movies[mask].sort_values(by='popularity', ascending=False).head(5)
            
            for row in recs.itertuples():
                c1, c2 = st.columns([1, 4])
                c1.image(fetch_poster(row.movie_id))
                c2.subheader(row.title)
                c2.write(get_groq_explanation(row.title, row.tags, mood))
                st.divider()

    # --- SIMILAR MOVIE SECTION ---
    elif st.session_state.page == 'movie':
        selected = st.selectbox("Pick a movie:", movies['title'].values)
        idx = movies[movies['title'] == selected].index[0]
        distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])
        
        # Slicing logic for "Suggest Next"
        start = st.session_state.recommend_offset
        end = start + 5
        current_recs = distances[start:end]

        st.write(f"### Suggestions based on '{selected}':")
        cols = st.columns(5)
        for i, dist in enumerate(current_recs):
            with cols[i]:
                m = movies.iloc[dist[0]]
                st.image(fetch_poster(m.movie_id))
                st.caption(m.title)
        
        # "Suggest Next" Button
        if st.button("Already seen these? Suggest next 🍿"):
            st.session_state.recommend_offset += 5
            st.rerun()

