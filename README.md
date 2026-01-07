# 🎬 CinemaAI: Emotion-Based Movie Recommender

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-f5f5f5?style=for-the-badge&logo=groq&logoColor=black)
![TMDB](https://img.shields.io/badge/TMDB-01d277?style=for-the-badge&logo=the-movie-database&logoColor=white)

**CinemaAI** is an end-to-end AI application that detects your real-time mood using a custom-trained **CNN** and provides personalized movie recommendations using **Groq AI (Llama 3)** for intelligent reasoning.

---

## 🚀 Key Features

* **🎭 Mood Detection**: Real-time facial emotion recognition (Happy, Sad, Angry, etc.) via webcam.
* **🧠 AI Explanations**: Uses **Groq Cloud** to explain exactly *why* a movie matches your detected mood.
* **🍿 Similar Movie Search**: A content-based engine that finds movies similar to your favorites.
* **🖼️ Dynamic Posters**: Fetches high-quality movie art directly from the **TMDB API**.

---

## 📸 Screenshots

| Emotion-Based-Movie Prediction | Movie Recommendations |
| :---: | :---: |
| ![Emotion-Based-Movie Prediction](<img width="1118" height="1004" alt="Screenshot 2026-01-07 185954" src="https://github.com/user-attachments/assets/1fd4952a-da8e-4e27-a059-290cc1a500a9" />
<img width="1846" height="1363" alt="Screenshot 2026-01-07 182326" src="https://github.com/user-attachments/assets/6b777c67-6f18-4df4-83e5-fa1ba1e4cd27" />
) | ![Similar Movies](<img width="1126" height="587" alt="Screenshot 2026-01-07 190043" src="https://github.com/user-attachments/assets/edeba59a-346a-48cc-a255-1b33558bd981" />
<img width="1846" height="1363" alt="Screenshot 2026-01-07 182326" src="https://github.com/user-attachments/assets/b57fc120-d2cc-46e3-b53b-256c4650d34f" />
) |

---

## 🛠️ Project Structure

The repository is organized for clear development and easy deployment:
* `app.py`: The main Streamlit dashboard.
* `Model-Files/`: Contains the trained `.keras` model and similarity data.
* `Images/`: Validation metrics and project graphics.
* `scripts/`: Original notebooks used for model training and testing.

---

## 💻 Getting Started

### 1. Prerequisites
Ensure you have Python installed and get your API keys from:
* [Groq Cloud Console](https://console.groq.com/)
* [TMDB API](https://www.themoviedb.org/documentation/api)

### 2. Installation
```bash
# Clone the repository
git clone [https://github.com/Zalanemoj/Movie-Recommendation-Based-On-Emotions.git](https://github.com/Zalanemoj/Movie-Recommendation-Based-On-Emotions.git)

# Install dependencies
pip install -r requirements.txt
