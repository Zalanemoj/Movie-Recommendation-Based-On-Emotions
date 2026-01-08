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

# --- PAGE CONFIG (MUST BE FIRST) ---
st.set_page_config(
    page_title="🎬 CinemaAI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS FOR MODERN UI ---
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    /* Title styling */
    .main-title {
        text-align: center;
        font-size: 4rem;
        font-weight: 700;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        text-shadow: 0 0 30px rgba(255,107,107,0.5);
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #a0a0a0;
        margin-bottom: 3rem;
    }
    
    /* Card styling */
    .option-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
        cursor: pointer;
        height: 100%;
    }
    
    .option-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 107, 107, 0.5);
        background: rgba(255, 255, 255, 0.08);
    }
    
    .option-icon {
        font-size: 5rem;
        margin-bottom: 1rem;
        filter: drop-shadow(0 0 20px rgba(255,107,107,0.5));
    }
    
    .option-title {
        font-size: 2rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 0.5rem;
    }
    
    .option-description {
        font-size: 1rem;
        color: #b0b0b0;
        line-height: 1.6;
    }
    
    /* Movie card styling */
    .movie-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 2rem;
        transition: all 0.3s ease;
    }
    
    .movie-card:hover {
        transform: scale(1.02);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }
    
    .movie-title {
        font-size: 1.8rem;
        font-weight: 600;
        color: #FF6B6B;
        margin-bottom: 1rem;
    }
    
    .movie-explanation {
        font-size: 1rem;
        color: #d0d0d0;
        line-height: 1.8;
    }
    
    /* Mood badge */
    .mood-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        border-radius: 25px;
        font-size: 1.2rem;
        font-weight: 600;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(255, 107, 107, 0.4);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(255, 107, 107, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(255, 107, 107, 0.5);
    }
    
    /* File uploader */
    .stFileUploader {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 2rem;
        border: 2px dashed rgba(255, 255, 255, 0.2);
    }
    
    /* Select box */
    .stSelectbox {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
    }
    
    /* Loading spinner */
    .stSpinner > div {
        border-top-color: #FF6B6B !important;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: rgba(0, 0, 0, 0.3);
    }
    
    /* Movie poster container */
    .movie-poster {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        transition: all 0.3s ease;
    }
    
    .movie-poster:hover {
        transform: scale(1.05);
        box-shadow: 0 15px 40px rgba(255, 107, 107, 0.4);
    }
    
    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(255, 107, 107, 0.5), transparent);
        margin: 2rem 0;
    }
    
    /* Back button */
    .back-button {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    
    .back-button:hover {
        background: rgba(255, 255, 255, 0.15) !important;
    }
</style>
""", unsafe_allow_html=True)

# --- AUTHENTICATION & SECRETS ---
TMDB_API_KEY = st.secrets["TMDB_API_KEY"] 
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
os.environ['KAGGLE_USERNAME'] = st.secrets["KAGGLE_USERNAME"]
os.environ['KAGGLE_KEY'] = st.secrets["KAGGLE_KEY"]

client = Groq(api_key=GROQ_API_KEY)

# Configuration
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

EMOTION_EMOJIS = {
    'happy': '😊',
    'sad': '😢',
    'angry': '😠',
    'fear': '😨',
    'surprise': '😲',
    'neutral': '😐',
    'disgust': '🤢'
}

# --- LOADING ASSETS ---
@st.cache_resource
def load_assets():
    """Downloads model from Kaggle and loads data files."""
    model_handle = "krishnapalsinhzala13/emotion-detector/keras/default"
    
    try:
        with st.spinner("🎬 Loading CinemaAI..."):
            model_dir = kagglehub.model_download(model_handle)
            
            # Search for any .keras file in the directory
            from pathlib import Path
            keras_files = list(Path(model_dir).rglob("*.keras"))
            
            if keras_files:
                model = load_model(str(keras_files[0]))
            else:
                # Fallback to specific filename
                model = load_model(os.path.join(model_dir, 'best_model (1).keras'))
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()
    
    try:
        movies = pd.DataFrame(pickle.load(open('Model-Files/movie_dict.pkl', 'rb')))
        similarity = pickle.load(open('Model-Files/similarity_matrix.pkl', 'rb'))
    except FileNotFoundError as e:
        st.error(f"Could not find data files: {e}")
        st.stop()
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    return model, movies, similarity, face_cascade

model, movies, similarity, face_cascade = load_assets()

# --- HELPER FUNCTIONS ---
def fetch_poster(movie_id):
    """Retrieves movie poster URL from TMDB API."""
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
    try:
        data = requests.get(url).json()
        return "https://image.tmdb.org/t/p/w500/" + data['poster_path']
    except:
        return "https://via.placeholder.com/500x750?text=No+Poster"

def get_groq_explanation(movie_title, movie_tags, emotion):
    """Uses Groq to generate match reasons."""
    prompt = (f"User is feeling {emotion}. Explain in 2-3 engaging sentences why '{movie_title}' "
              f"is the perfect movie for their mood. Use these keywords: {movie_tags}. "
              f"Be enthusiastic and persuasive!")
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8, max_tokens=200
        )
        return completion.choices[0].message.content
    except:
        return f"This {emotion} movie matches your current vibe perfectly! 🎬"

def detect_and_predict(image_frame):
    """Predicts emotion using the model."""
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 3)
    
    roi = gray[faces[0][1]:faces[0][1]+faces[0][3], faces[0][0]:faces[0][0]+faces[0][2]] if len(faces) > 0 else gray
    roi = cv2.resize(roi, (48, 48))
    roi = roi.astype('float') / 255.0
    roi = np.expand_dims(img_to_array(roi), axis=0)
    
    preds = model.predict(roi, verbose=0)
    return EMOTIONS[np.argmax(preds)]

# --- STATE MANAGEMENT ---
if 'page' not in st.session_state: 
    st.session_state.page = 'home'
if 'recommend_offset' not in st.session_state: 
    st.session_state.recommend_offset = 1

# --- MAIN APP ---

# Home Page
if st.session_state.page == 'home':
    # Title
    st.markdown('<h1 class="main-title">🎬 CinemaAI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Your AI-Powered Movie Companion • Discover Films That Match Your Mood</p>', unsafe_allow_html=True)
    
    # Add some spacing
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Two columns for options
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
        <div class="option-card">
            <div class="option-icon">🎭</div>
            <div class="option-title">Mood-Based</div>
            <div class="option-description">
                Upload your photo and let AI detect your emotion,
                then get personalized movie recommendations that match your vibe
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🎭 Start Mood Detection", use_container_width=True, key="mood_btn"):
            st.session_state.page = 'mood'
            st.rerun()
    
    with col2:
        st.markdown("""
        <div class="option-card">
            <div class="option-icon">🍿</div>
            <div class="option-title">Similar Movies</div>
            <div class="option-description">
                Find movies similar to your favorites using our advanced
                recommendation algorithm trained on thousands of films
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🍿 Find Similar Movies", use_container_width=True, key="similar_btn"):
            st.session_state.page = 'movie'
            st.session_state.recommend_offset = 1
            st.rerun()
    
    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: #808080; font-size: 0.9rem;">
        Powered by AI • Groq • TMDB • Made with ❤️
    </div>
    """, unsafe_allow_html=True)

# Mood-Based Page
elif st.session_state.page == 'mood':
    # Back button
    if st.button("⬅ Back to Home", key="back_mood"):
        st.session_state.page = 'home'
        st.rerun()
    
    st.markdown('<h1 class="main-title">🎭 Mood Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Upload your photo and let AI work its magic</p>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Centered file uploader
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        img_file = st.file_uploader(
            "📸 Choose an image", 
            type=['jpg', 'png', 'jpeg'],
            help="Upload a clear photo of your face for best results"
        )
    
    if img_file:
        # Display uploaded image
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(img_file, caption="Your Photo", use_container_width=True)
        
        # Process image
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        with st.spinner("🔍 Analyzing your emotion..."):
            mood = detect_and_predict(image)
        
        # Display detected mood
        emoji = EMOTION_EMOJIS.get(mood, '😊')
        st.markdown(f"""
        <div style="text-align: center; margin: 2rem 0;">
            <div style="font-size: 5rem;">{emoji}</div>
            <div class="mood-badge">Detected Mood: {mood.upper()}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Get recommendations
        target_genres = EMOTION_TO_GENRE.get(mood, [])
        mask = movies['genres'].apply(lambda x: any(g in x for g in target_genres))
        recs = movies[mask].sort_values(by='popularity', ascending=False).head(5)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<h2 style="text-align: center; color: #FF6B6B;">Perfect Movies for Your {mood.title()} Mood</h2>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Display recommendations
        for idx, row in enumerate(recs.itertuples(), 1):
            col1, col2 = st.columns([1, 2], gap="large")
            
            with col1:
                st.markdown('<div class="movie-poster">', unsafe_allow_html=True)
                st.image(fetch_poster(row.movie_id), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown(f'<div class="movie-title">#{idx} {row.title}</div>', unsafe_allow_html=True)
                explanation = get_groq_explanation(row.title, row.tags, mood)
                st.markdown(f'<div class="movie-explanation">{explanation}</div>', unsafe_allow_html=True)
            
            st.markdown("<hr>", unsafe_allow_html=True)

# Similar Movies Page
elif st.session_state.page == 'movie':
    # Back button
    if st.button("⬅ Back to Home", key="back_similar"):
        st.session_state.page = 'home'
        st.rerun()
    
    st.markdown('<h1 class="main-title">🍿 Similar Movies</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Discover movies you\'ll love based on your favorites</p>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Movie selection
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        selected = st.selectbox(
            "🎬 Choose a movie you like:",
            movies['title'].values,
            help="Select a movie to get similar recommendations"
        )
    
    if selected:
        idx = movies[movies['title'] == selected].index[0]
        distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])
        
        # Slicing logic
        start = st.session_state.recommend_offset
        end = st.session_state.recommend_offset + 5
        current_recs = distances[start:end]
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown(f'<h2 style="text-align: center; color: #4ECDC4;">Movies Similar to "{selected}"</h2>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Display in columns
        cols = st.columns(5, gap="medium")
        for i, dist in enumerate(current_recs):
            with cols[i]:
                m = movies.iloc[dist[0]]
                st.markdown('<div class="movie-poster">', unsafe_allow_html=True)
                st.image(fetch_poster(m.movie_id), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown(f'<div style="text-align: center; color: white; font-weight: 600; margin-top: 1rem;">{m.title}</div>', unsafe_allow_html=True)
                match_percent = int(dist[1] * 100)
                st.markdown(f'<div style="text-align: center; color: #4ECDC4; font-size: 0.9rem;">Match: {match_percent}%</div>', unsafe_allow_html=True)
        
        # Next button
        st.markdown("<br><br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🎬 Show Me More Movies", use_container_width=True):
                st.session_state.recommend_offset += 5
                st.rerun()
