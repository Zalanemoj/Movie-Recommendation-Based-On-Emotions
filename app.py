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

# --- 2. ASSET LOADING ---
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
    preds = model.predict(roi, verbose=0)
    return EMOTIONS[np.argmax(preds)]

# --- 4. STATE MANAGEMENT ---
# Initialize session state variables if they don't exist
if 'page' not in st.session_state: st.session_state.page = 'home'

# State for Movie Recommender
if 'movie_offset' not in st.session_state: st.session_state.movie_offset = 0
if 'show_movie_recs' not in st.session_state: st.session_state.show_movie_recs = False
if 'last_selected_movie' not in st.session_state: st.session_state.last_selected_movie = ""

# State for Mood Recommender
if 'mood_offset' not in st.session_state: st.session_state.mood_offset = 0
if 'last_img_bytes' not in st.session_state: st.session_state.last_img_bytes = None

st.set_page_config(page_title="CinemaAI", layout="wide")
st.title("🎬 CinemaAI: Personalized Recommender")

# --- 5. NAVIGATION ---
if st.session_state.page == 'home':
    st.subheader("Welcome to your Personalized Cinema Experience!")
    st.markdown("""
    ### 🌟 How to use this App:
    1. **🎭 Mood-Based Movie**: We scan your face to detect emotion and suggest movies. **Click 'Show Next' to see more options.**
    2. **🍿 Only Movie Recommender**: Select a movie you love. We'll show similar ones. **Pagination is now fixed!**
    3. **🧠 Only Emotion Detection**: Detect your mood and get AI wellness tips.
    """)
    col1, col2, col3 = st.columns(3)
    if col1.button("🎭 Mood-Based Movie", use_container_width=True): st.session_state.page = 'mood_movie'; st.rerun()
    if col2.button("🍿 Only Movie Recommender", use_container_width=True): st.session_state.page = 'only_movie'; st.rerun()
    if col3.button("🧠 Only Emotion Detection", use_container_width=True): st.session_state.page = 'only_emotion'; st.rerun()

else:
    if st.sidebar.button("⬅ Back to Menu"):
        # Reset states when going back
        st.session_state.show_movie_recs = False
        st.session_state.movie_offset = 0
        st.session_state.mood_offset = 0
        st.session_state.last_img_bytes = None
        st.session_state.page = 'home'
        st.rerun()

    # --- MODE 1: MOOD-BASED MOVIE ---
    if st.session_state.page == 'mood_movie':
        st.header("Movie Recommendations Based on Your Emotion")
        
        tab1, tab2 = st.tabs(["📸 Use Camera", "📁 Upload Image"])
        with tab1: cam_file = st.camera_input("Capture face")
        with tab2: up_file = st.file_uploader("Upload image", type=['jpg', 'png', 'jpeg'])
        
        img_file = cam_file if cam_file else up_file

        if img_file:
            # Convert file to bytes to check if it's a new image
            file_bytes_obj = img_file.getvalue()
            
            # If image changed, reset offset to 0
            if st.session_state.last_img_bytes != file_bytes_obj:
                st.session_state.last_img_bytes = file_bytes_obj
                st.session_state.mood_offset = 0

            # Decode and Predict
            file_bytes = np.asarray(bytearray(file_bytes_obj), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            mood = detect_and_predict(image)
            st.success(f"Detected Mood: **{mood.upper()}**")
            
            # Get ALL recommendations (sorted by popularity), not just head(5)
            target_genres = EMOTION_TO_GENRE.get(mood, [])
            mask = movies['genres'].apply(lambda x: any(g in x for g in target_genres))
            all_recs = movies[mask].sort_values(by='popularity', ascending=False)
            
            # Pagination Logic
            start = st.session_state.mood_offset
            end = start + 5
            batch_recs = all_recs.iloc[start:end]
            
            if batch_recs.empty:
                st.warning("No more movies to show for this mood!")
            else:
                st.subheader(f"Movies for a {mood} day (Showing {start+1}-{end}):")
                for row in batch_recs.itertuples():
                    c1, c2 = st.columns([1, 4])
                    with c1: st.image(fetch_poster(row.movie_id))
                    with c2:
                        st.subheader(row.title)
                        with st.spinner("AI is thinking..."):
                            st.write(get_groq_explanation(row.title, row.tags, mood))
                    st.divider()

                # Next Button
                if st.button("Already seen these? Show Next 5 ➡️", key="next_mood_btn"):
                    st.session_state.mood_offset += 5
                    st.rerun()

    # --- MODE 2: ONLY MOVIE RECOMMENDER ---
    elif st.session_state.page == 'only_movie':
        st.header("Similar Movie Recommender")
        
        selected = st.selectbox("Pick a movie you liked:", movies['title'].values)
        
        # Reset if user picks a completely new movie from dropdown
        if selected != st.session_state.last_selected_movie:
            st.session_state.show_movie_recs = False
            st.session_state.movie_offset = 0
            st.session_state.last_selected_movie = selected

        # When "Get Recommendations" is clicked, we LOCK the state to True
        if st.button("Get Recommendations"):
            st.session_state.show_movie_recs = True
            st.session_state.movie_offset = 0  # Start from beginning
        
        # Only show if the state is True
        if st.session_state.show_movie_recs:
            idx = movies[movies['title'] == selected].index[0]
            distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])
            
            # Pagination Logic
            start = st.session_state.movie_offset
            # Skip index 0 because it's the movie itself
            # If offset is 0, we want 1:6. If offset is 5, we want 6:11.
            # Logic: start index is 1 + offset.
            slice_start = 1 + start
            slice_end = slice_start + 5
            
            current_batch = distances[slice_start:slice_end]
            
            cols = st.columns(5)
            for i, dist in enumerate(current_batch):
                with cols[i]:
                    m = movies.iloc[dist[0]]
                    st.image(fetch_poster(m.movie_id))
                    st.caption(m.title)
            
            st.divider()
            
            # Next Button
            if st.button("Already seen these? Show Next 5 ➡️", key="next_movie_btn"):
                st.session_state.movie_offset += 5
                st.rerun()

    # --- MODE 3: ONLY EMOTION DETECTION ---
    elif st.session_state.page == 'only_emotion':
        st.header("Emotion Detection & AI Tips")
        
        tab1, tab2 = st.tabs(["📸 Use Camera", "📁 Upload Image"])
        with tab1: cam_file = st.camera_input("Capture face for analysis")
        with tab2: up_file = st.file_uploader("Upload face image", type=['jpg', 'png', 'jpeg'])
        
        img_file = cam_file if cam_file else up_file

        if img_file:
            file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            mood = detect_and_predict(image)
            
            c1, c2 = st.columns([1, 2])
            with c1: st.image(image, channels="BGR", caption="Input Image", width=300)
            with c2:
                st.metric(label="Predicted Mood", value=mood.upper())
                st.markdown("### 💡 AI Wellness Tips")
                with st.spinner("Generating tips..."):
                    st.info(get_mood_tips(mood))

