import pandas as pd
import numpy as np
import pickle
import cv2
import time
import argparse
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# --- 1. SETTINGS & ARGUMENTS ---
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help="Path to a specific image file")
args = parser.parse_args()

# --- 2. LOAD AI ASSETS ---
print("Loading AI models and movie data... please wait.")
# Ensure these files are in the same folder as this script
model = load_model('best_model.keras')
movies = pd.DataFrame(pickle.load(open('movie_dict.pkl', 'rb')))
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define standard labels and genre mappings
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


# --- 3. CORE LOGIC FUNCTIONS ---

def get_mood_recommendations(emotion):
    """Filters the movie database based on the detected emotion."""
    target_genres = EMOTION_TO_GENRE.get(emotion, [])

    # Filter by genre (Checks if any target genre exists in the movie's genre list)
    mask = movies['genres'].apply(lambda x: any(genre in x for genre in target_genres))
    filtered_movies = movies[mask].copy()

    if filtered_movies.empty:
        return ["No movies found for this mood in your current dataset."]

    # Sort by popularity to show the best matches first
    # This requires 'popularity' to be in your movie_dict.pkl
    return filtered_movies.sort_values(by='popularity', ascending=False).head(5)['title'].tolist()


def detect_and_predict(image_frame):
    """Detects face, preprocesses image, and returns the predicted emotion label."""
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

    # Try detecting the face with relaxed parameters
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        roi = gray[y:y + h, x:x + w]
    else:
        # FALLBACK: If face detection fails (common in validation sets), use full image
        roi = gray

    # Standard Preprocessing: Resize to 48x48 and normalize
    roi = cv2.resize(roi, (48, 48))
    roi = roi.astype('float') / 255.0
    roi = np.expand_dims(img_to_array(roi), axis=0)

    # Predict
    preds = model.predict(roi, verbose=0)
    return EMOTIONS[np.argmax(preds)]


# --- 4. EXECUTION FLOW ---

detected_emotion = None

# MODE A: Image Path Provided
if args.path:
    if os.path.isfile(args.path):
        img = cv2.imread(args.path)
        if img is not None:
            print(f"\n--- Analyzing Image: {args.path} ---")
            detected_emotion = detect_and_predict(img)
        else:
            print(f"Error: Could not read image at {args.path}")
    else:
        print(f"Error: File not found at {args.path}")

# MODE B: No Path Provided -> Turn on Webcam
else:
    print("\n--- No path provided. Starting 5-second Webcam capture ---")
    cap = cv2.VideoCapture(0)
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret: break

        elapsed = int(time.time() - start_time)
        detected_emotion = detect_and_predict(frame)

        # Display feedback on the webcam window
        cv2.putText(frame, f"Time left: {5 - elapsed}s", (20, 40), 1, 2, (0, 0, 255), 2)
        cv2.putText(frame, f"Model sees: {detected_emotion.upper()}", (20, 80), 1, 2, (0, 255, 0), 2)
        cv2.imshow('Mood Analysis (Press Q to quit)', frame)

        if elapsed >= 5 or cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# --- 5. FINAL RESULTS ---

if detected_emotion:
    print("\n" + "=" * 40)
    print(f" FINAL DETECTED MOOD: {detected_emotion.upper()}")
    print("=" * 40)

    recommendations = get_mood_recommendations(detected_emotion)
    print("\nTop Movie Recommendations for you:")
    for i, title in enumerate(recommendations, 1):
        print(f"{i}. {title}")
else:
    print("\n[!] Processing failed. No emotion detected.")