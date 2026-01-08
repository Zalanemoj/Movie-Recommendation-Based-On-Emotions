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
# Ensure these keys are added in your Streamlit Cloud "Secrets" dashboard
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

# --- 2. ASSET LOADING (KAGGLE MODEL + GITHUB DATA) ---
@st.cache_resource
def load_assets():
    # 1. Download the model from Kaggle Registry
    model_handle = "krishnapalsinhzala13/emotion-detector/keras/default"
    with st.spinner("Fetching model from Kaggle..."):
        model_dir = kagglehub.model_download(model_handle)
    
    # 2. Load model from the Kaggle directory
    model = load_model(os.path.join(model_dir, 'best_model (1).keras'))
    
    # 3. Load supporting data files from the GitHub repo's 'Model-Files' folder
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
    """Generates AI tips to change or manage the detected mood."""
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
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 3)
    roi = gray[faces[0][1]:faces[0][1]+faces[0][3], faces[0][0]:faces[0][0]+faces[0][2]] if len(faces) > 0 else gray
    roi = cv2.resize(roi, (48, 48))
    roi = roi.astype('float') / 255.0
    roi = np.expand_dims(img_to_array(roi), axis=0)
    preds = model.predict(roi, verbose=0)
    return EMOTIONS[np.argmax(preds)]

# --- 4. STATE MANAGEMENT ---
if 'page' not in st.session_state: st.session_state.page = 'home'
if 'recommend_offset' not in st.session_state: st.session_state.recommend_offset = 1

st.set_page_config(page_title="CinemaAI", layout="wide")
st.title("🎬 CinemaAI: Personalized Recommender")

# --- 5. NAVIGATION ---
if st.session_state.page == 'home':
    st.subheader("Welcome to your Personalized Cinema Experience!")
    
    # Description/Instructions
    st.markdown("""
    ### How to use this App:
    1. **🎭 Mood-Based Movie**: Detect your current emotion via camera or upload, and we'll suggest movies that fit your vibe perfectly.
    2. **🍿 Only Movie Recommender**: Already have a favorite movie? Select it to find similar titles, and use 'Suggest Next' if you've seen them all.
    3. **🧠 Only Emotion Detection**: Just curious about your mood? Detect your emotion and get AI-powered tips on how to improve or manage it.
    """)
    
    c1, c2, c3 = st.columns(3)
    if c1.button("🎭 Mood-Based Movie", use_container_width=True):
        st.session_state.page = 'mood_movie'; st.rerun()
    if c2.button("🍿 Only Movie Recommender", use_container_width=True):
        st.session_state.page = 'only_movie'; st.session_state.recommend_offset = 1; st.rerun()
    if c3.button("🧠 Only Emotion Detection", use_container_width=True):
        st.session_state.page = 'only_emotion'; st.rerun()

else:
    if st.sidebar.button("⬅ Back to Menu"):
        st.session_state.page = 'home'; st.rerun()

    # --- MODE 1: MOOD-BASED MOVIE ---
    if st.session_state.page == 'mood_movie':
        st.header("Movie Recommendations Based on Your Emotion")
        img_file = st.file_uploader("Capture or Upload Face", type=['jpg', 'png'])
        if img_file:
            file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            mood = detect_and_predict(image)
            st.success(f"Detected Mood: {mood.upper()}")
            
            target_genres = EMOTION_TO_GENRE.get(mood, [])
            mask = movies['genres'].apply(lambda x: any(g in x for g in target_genres))
            recs = movies[mask].sort_values(by='popularity', ascending=False).head(5)
            
            for row in recs.itertuples():
                col1, col2 = st.columns([1, 4])
                col1.image(fetch_poster(row.movie_id))
                col2.subheader(row.title)
                col2.write(get_groq_explanation(row.title, row.tags, mood))
                st.divider()

    # --- MODE 2: ONLY MOVIE RECOMMENDER ---
    elif st.session_state.page == 'only_movie':
        st.header("Similar Movie Recommender")
        selected = st.selectbox("Pick a movie you liked:", movies['title'].values)
        idx = movies[movies['title'] == selected].index[0]
        distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])
        
        # Slicing logic for "Suggest Next"
        start = st.session_state.recommend_offset
        end = start + 5
        current_recs = distances[start:end]

        cols = st.columns(5)
        for i, dist in enumerate(current_recs):
            with cols[i]:
                m = movies.iloc[dist[0]]
                st.image(fetch_poster(m.movie_id))
                st.caption(m.title)
        
        if st.button("Already seen these? Suggest next 🍿"):
            st.session_state.recommend_offset += 5; st.rerun()

    # --- MODE 3: ONLY EMOTION DETECTION ---
    elif st.session_state.page == 'only_emotion':
        st.header("Emotion Detection & AI Mood Tips")
        img_file = st.file_uploader("Check your Mood", type=['jpg', 'png'])
        if img_file:
            file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            mood = detect_and_predict(image)
            st.metric(label="Predicted Mood", value=mood.upper())
            
            st.write("### AI Tips to manage/change your mood:")
            st.info(get_mood_tips(mood))
