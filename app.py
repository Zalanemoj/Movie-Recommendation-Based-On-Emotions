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
# Streamlit Cloud reads these from your 'Secrets' settings.
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
    """
    Downloads model from Kaggle Registry and loads pickles from the local repo.
    """
    # 1. Download the model from Kaggle
    # Using your specific Kaggle Model path
    model_handle = "krishnapalsinhzala13/emotion-detector/keras/default"
    
    with st.spinner("Downloading model from Kaggle Registry..."):
        # kagglehub returns the path to the downloaded model folder
        model_dir = kagglehub.model_download(model_handle)
    
    # 2. Load the model from the Kaggle download directory
    # This ignores any model file in your GitHub repo
    model = load_model(os.path.join(model_dir, 'best_model.keras'))
    
    # 3. Load pickle data files directly from your GitHub repo folder
    # These are located in 'Model-Files/' within your repository
    movies = pd.DataFrame(pickle.load(open('Model-Files/movie_dict.pkl', 'rb')))
    similarity = pickle.load(open('Model-Files/similarity_matrix.pkl', 'rb'))
    
    # Load OpenCV's face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    return model, movies, similarity, face_cascade

# Initialize all assets
model, movies, similarity, face_cascade = load_assets()

# --- STEP 3: HELPER FUNCTIONS ---
def fetch_poster(movie_id):
    """Retrieves movie poster URL from TMDB API."""
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
    try:
        data = requests.get(url).json()
        return "https://image.tmdb.org/t/p/w500/" + data['poster_path']
    except:
        return "https://via.placeholder.com/500x750?text=No+Poster"

def get_groq_explanation(movie_title, movie_tags, emotion):
    """Uses Groq Llama-3 to generate match reasons."""
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
        return f"This {emotion} movie matches your current vibe perfectly!"

def detect_and_predict(image_frame):
    """Predicts emotion using the Kaggle-loaded model."""
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
    st.subheader("Choose your recommendation method:")
    col1, col2 = st.columns(2)
    if col1.button("🎭 Mood-Based", use_container_width=True):
        st.session_state.page = 'mood'; st.rerun()
    if col2.button("🍿 Similar-Movie", use_container_width=True):
        st.session_state.page = 'movie'; st.session_state.recommend_offset = 1; st.rerun()

else:
    if st.sidebar.button("⬅ Back to Menu"):
        st.session_state.page = 'home'; st.rerun()

    if st.session_state.page == 'mood':
        img_file = st.file_uploader("Upload or Capture face", type=['jpg', 'png'])
        if img_file:
            file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            mood = detect_and_predict(image)
            st.success(f"Detected Mood: {mood.upper()}")
            
            target_genres = EMOTION_TO_GENRE.get(mood, [])
            mask = movies['genres'].apply(lambda x: any(g in x for g in target_genres))
            recs = movies[mask].sort_values(by='popularity', ascending=False).head(5)
            
            for row in recs.itertuples():
                c1, c2 = st.columns([1, 4])
                c1.image(fetch_poster(row.movie_id))
                c2.subheader(row.title)
                c2.write(get_groq_explanation(row.title, row.tags, mood))
                st.divider()

    elif st.session_state.page == 'movie':
        selected = st.selectbox("Pick a movie:", movies['title'].values)
        idx = movies[movies['title'] == selected].index[0]
        distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])
        
        # Slicing logic for "Suggest Next"
        start, end = st.session_state.recommend_offset, st.session_state.recommend_offset + 5
        current_recs = distances[start:end]

        cols = st.columns(5)
        for i, dist in enumerate(current_recs):
            with cols[i]:
                m = movies.iloc[dist[0]]
                st.image(fetch_poster(m.movie_id))
                st.caption(m.title)
        
        if st.button("Already seen these? Suggest next 🍿"):
            st.session_state.recommend_offset += 5; st.rerun()
