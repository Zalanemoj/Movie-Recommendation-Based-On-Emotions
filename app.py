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

# --- 1. CONFIGURATION & SECRETS ---
# Ensure these keys are in your .streamlit/secrets.toml file
TMDB_API_KEY = st.secrets["TMDB_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# Set Kaggle credentials for kagglehub
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

# --- 2. ASSET LOADING (KAGGLE + GITHUB) ---
@st.cache_resource
def load_assets():
    # 1. Download model from Kaggle
    model_handle = "krishnapalsinhzala13/emotion-detector/keras/default"
    with st.spinner("Downloading Model from Kaggle..."):
        model_dir = kagglehub.model_download(model_handle)
    
    # 2. Load the Keras model
    model = load_model(os.path.join(model_dir, 'best_model.keras'))
    
    # 3. Load DataFrames (ensure Model-Files folder exists in repo)
    movies = pd.DataFrame(pickle.load(open('Model-Files/movie_dict.pkl', 'rb')))
    similarity = pickle.load(open('Model-Files/similarity_matrix.pkl', 'rb'))
    
    # 4. Load Face Cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    return model, movies, similarity, face_cascade

model, movies, similarity, face_cascade = load_assets()

# --- 3. HELPER FUNCTIONS ---
def fetch_poster(movie_id):
    """Fetches movie poster from TMDB API."""
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
    try:
        data = requests.get(url).json()
        return "https://image.tmdb.org/t/p/w500/" + data['poster_path']
    except:
        return "https://via.placeholder.com/500x750?text=No+Poster"

def get_groq_explanation(movie_title, movie_tags, emotion):
    """Uses Groq to explain why a movie fits the mood."""
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
    """Generates 3 practical tips to manage mood."""
    prompt = f"A user is feeling {emotion}. Provide 3 short, practical tips to improve or manage this mood effectively."
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7, max_tokens=150
        )
        return completion.choices[0].message.content
    except:
        return "Take a deep breath and stay positive!"

def detect_and_predict(image_frame):
    """Detects face and predicts emotion."""
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 3)
    # Crop face if detected, else use whole image
    roi = gray[faces[0][1]:faces[0][1]+faces[0][3], faces[0][0]:faces[0][0]+faces[0][2]] if len(faces) > 0 else gray
    roi = cv2.resize(roi, (48, 48))
    roi = roi.astype('float') / 255.0
    roi = np.expand_dims(img_to_array(roi), axis=0)
    preds = model.predict(roi, verbose=0)
    return EMOTIONS[np.argmax(preds)]

# --- 4. STATE MANAGEMENT ---
if 'page' not in st.session_state: st.session_state.page = 'home'
if 'recommend_offset' not in st.session_state: st.session_state.recommend_offset = 1
if 'last_selected_movie' not in st.session_state: st.session_state.last_selected_movie = ""

st.set_page_config(page_title="CinemaAI", layout="wide")
st.title("🎬 CinemaAI: Personalized Recommender")

# --- 5. NAVIGATION & PAGES ---

# === HOME PAGE ===
if st.session_state.page == 'home':
    st.subheader("Welcome to your Personalized Cinema Experience!")
    
    st.markdown("""
    ### 🌟 How to use this App:
    1. **🎭 Mood-Based Movie**: We scan your face (Camera or Upload) to detect your emotion and suggest movies that fit your vibe perfectly.
    2. **🍿 Only Movie Recommender**: Select a movie you love, and we'll recommend similar ones. **If you've seen them, click 'Next' for more!**
    3. **🧠 Only Emotion Detection**: Just want to check your mood? We'll detect it and give you AI-powered tips to manage it.
    """)
    
    col1, col2, col3 = st.columns(3)
    if col1.button("🎭 Mood-Based Movie", use_container_width=True):
        st.session_state.page = 'mood_movie'
        st.rerun()
    if col2.button("🍿 Only Movie Recommender", use_container_width=True):
        st.session_state.page = 'only_movie'
        st.rerun()
    if col3.button("🧠 Only Emotion Detection", use_container_width=True):
        st.session_state.page = 'only_emotion'
        st.rerun()

# === APP PAGES ===
else:
    if st.sidebar.button("⬅ Back to Menu"):
        st.session_state.page = 'home'
        st.rerun()

    # --- MODE 1: MOOD-BASED MOVIE ---
    if st.session_state.page == 'mood_movie':
        st.header("Movie Recommendations Based on Your Emotion")
        
        # Tabs for Camera vs Upload
        tab1, tab2 = st.tabs(["📸 Use Camera", "📁 Upload Image"])
        
        with tab1:
            cam_file = st.camera_input("Capture face")
        with tab2:
            up_file = st.file_uploader("Upload image", type=['jpg', 'png', 'jpeg'])
            
        # Determine which input to use
        img_file = cam_file if cam_file else up_file

        if img_file:
            file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            
            # Prediction
            mood = detect_and_predict(image)
            st.success(f"Detected Mood: **{mood.upper()}**")
            
            # Recommendation Logic
            target_genres = EMOTION_TO_GENRE.get(mood, [])
            mask = movies['genres'].apply(lambda x: any(g in x for g in target_genres))
            recs = movies[mask].sort_values(by='popularity', ascending=False).head(5)
            
            st.subheader(f"Movies for a {mood} day:")
            for row in recs.itertuples():
                c1, c2 = st.columns([1, 4])
                with c1:
                    st.image(fetch_poster(row.movie_id))
                with c2:
                    st.subheader(row.title)
                    with st.spinner("AI is thinking..."):
                        st.write(get_groq_explanation(row.title, row.tags, mood))
                st.divider()

    # --- MODE 2: ONLY MOVIE RECOMMENDER ---
    elif st.session_state.page == 'only_movie':
        st.header("Similar Movie Recommender")
        
        selected = st.selectbox("Pick a movie you liked:", movies['title'].values)
        
        # Reset offset if user picks a new movie
        if selected != st.session_state.last_selected_movie:
            st.session_state.recommend_offset = 1
            st.session_state.last_selected_movie = selected

        if st.button("Get Recommendations"):
            idx = movies[movies['title'] == selected].index[0]
            # Calculate distances
            distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])
            
            # Pagination Logic
            start = st.session_state.recommend_offset
            end = start + 5
            current_batch = distances[start:end]
            
            cols = st.columns(5)
            for i, dist in enumerate(current_batch):
                with cols[i]:
                    m = movies.iloc[dist[0]]
                    st.image(fetch_poster(m.movie_id))
                    st.caption(m.title)
            
            st.divider()
            # The "Next" Button
            if st.button("Already seen these? Show Next 5 ➡️"):
                st.session_state.recommend_offset += 5
                st.rerun()

    # --- MODE 3: ONLY EMOTION DETECTION ---
    elif st.session_state.page == 'only_emotion':
        st.header("Emotion Detection & AI Tips")
        
        # Tabs for Camera vs Upload
        tab1, tab2 = st.tabs(["📸 Use Camera", "📁 Upload Image"])
        
        with tab1:
            cam_file = st.camera_input("Capture face for analysis")
        with tab2:
            up_file = st.file_uploader("Upload face image", type=['jpg', 'png', 'jpeg'])
            
        img_file = cam_file if cam_file else up_file

        if img_file:
            file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            
            mood = detect_and_predict(image)
            
            c1, c2 = st.columns([1, 2])
            with c1:
                st.image(image, channels="BGR", caption="Input Image", width=300)
            with c2:
                st.metric(label="Predicted Mood", value=mood.upper())
                st.markdown("### 💡 AI Wellness Tips")
                with st.spinner("Generating tips..."):
                    st.info(get_mood_tips(mood))
