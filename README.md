# 🎬 CinemaAI: Emotion-Based Movie Recommender

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-f5f5f5?style=for-the-badge&logo=groq&logoColor=black)
![TMDB](https://img.shields.io/badge/TMDB-01d277?style=for-the-badge&logo=the-movie-database&logoColor=white)
![Banner](https://github.com/user-attachments/assets/4749ae32-e2f8-4842-80fd-6af80c187c71)

**CinemaAI** is an end-to-end AI application that detects your real-time mood using a custom-trained **CNN** and provides personalized movie recommendations using **Groq AI (Llama 3)** for intelligent reasoning.

---

## 🚀 Key Features

* **🎭 Mood Detection**: Real-time facial emotion recognition (Happy, Sad, Angry, etc.) via webcam.
* **🧠 AI Explanations**: Uses **Groq Cloud** to explain exactly *why* a movie matches your detected mood.
* **🍿 Similar Movie Search**: A content-based engine that finds movies similar to your favorites.
* **🖼️ Dynamic Posters**: Fetches high-quality movie art directly from the **TMDB API**.

---

## 📸 Screenshots
<img width="1118" height="1004" alt="Screenshot 2026-01-07 185954" src="https://github.com/user-attachments/assets/329e1276-0482-4ab7-a430-a8ba679d85ba" />


---

## 💻 Getting Started

### 1. Prerequisites
Ensure you have Python installed and get your API keys from:
* [Groq Cloud Console](https://console.groq.com/)
* [TMDB API](https://www.themoviedb.org/documentation/api)

### 2. Installation
bash
# Clone the repository
git clone [https://github.com/Zalanemoj/Movie-Recommendation-Based-On-Emotions.git](https://github.com/Zalanemoj/Movie-Recommendation-Based-On-Emotions.git)

# Install dependencies
pip install -r requirements.txt

### 3. 🛠️ 
Project Structure
The repository is organized for clear development and easy deployment:
* `app.py`: The main Streamlit dashboard.
* `Model-Files/`: Contains the trained `.keras` model and similarity data.
* `Images/`: Validation metrics and project graphics.
* `scripts/`: Original notebooks used for model training and testing.

Movie-Recommendation-Based-On-Emotions/
├── .idea/                          # IDE configuration files
│   ├── inspectionProfiles/         # Code inspection settings
│   ├── Emotion-Based-Movie-Recomender.iml
│   ├── modules.xml
│   └── ...
├── Images/                         # Project screenshots and confusion matrices
│   ├── Confusion-metrics-withClassWeights.png
│   ├── Confusion-metrics-withoutClassWeights.png
│   └── ... (sample movie posters/images)
├── Model-Files/                    # Pre-trained models and data files
│   ├── best_model.keras            # Trained emotion detection model
│   ├── movie_dict.pkl              # Pickled movie dictionary for recommendations
│   └── similarity_matrix.pkl       # Pre-computed similarity scores
├── scripts/                        # Jupyter notebooks and testing scripts
│   ├── content-based-movie-recomender.ipynb
│   ├── face-emotion-detection.ipynb
│   └── test_recommend.py           # Script for testing recommendation logic
├── app.py                          # Main Streamlit/Flask application file
├── requirements.txt                # Project dependencies
├── LICENSE                         # Project license
├── .gitignore                      # Files to exclude from Git
└── README.md                       # Project documentation
