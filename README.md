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

<img width="1118" alt="Screenshot 2026-01-07 185954" src="https://github.com/user-attachments/assets/329e1276-0482-4ab7-a430-a8ba679d85ba" />

---

## 💻 Getting Started

### 1. Prerequisites

Ensure you have Python installed and get your API keys from:
* [Groq Cloud Console](https://console.groq.com/)
* [TMDB API](https://www.themoviedb.org/documentation/api)

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/Zalanemoj/Movie-Recommendation-Based-On-Emotions.git

# Navigate to project directory
cd Movie-Recommendation-Based-On-Emotions

# Install dependencies
pip install -r requirements.txt
```

### 3. 🛠️ Project Structure

The repository is organized for clear development and easy deployment:
```
Movie-Recommendation-Based-On-Emotions/
│
├── .idea/                                    # IDE configuration files
│   ├── inspectionProfiles/
│   ├── Emotion-Based-Movie-Recomender.iml
│   ├── modules.xml
│   └── ...
│
├── Images/                                   # Validation metrics and graphics
│   ├── Confusion-metrics-withClassWeights.png
│   ├── Confusion-metrics-withoutClassWeights.png
│   └── ... (sample movie posters/images)
│
├── Model-Files/                              # Trained models and data
│   ├── best_model.keras                     # Emotion detection CNN model
│   ├── movie_dict.pkl                       # Movie metadata dictionary
│   └── similarity_matrix.pkl                # Content-based similarity matrix
│
├── scripts/                                  # Training and testing notebooks
│   ├── content-based-movie-recomender.ipynb # Movie recommendation system
│   ├── face-emotion-detection.ipynb         # Emotion detection model training
│   └── test_recommend.py                    # Testing script
│
├── app.py                                    # Main Streamlit application
├── requirements.txt                          # Python dependencies
├── LICENSE                                   # Project license
├── .gitignore                               # Git ignore rules
└── README.md                                # Project documentation
```

---

## 🎯 Usage

1. **Set up your API keys** in the application settings or environment variables
2. **Run the Streamlit app**:
```bash
   streamlit run app.py
```
3. **Allow webcam access** for real-time emotion detection
4. **Get personalized recommendations** based on your current mood!

---

## 🧪 Model Performance

The emotion detection model was trained on facial expression datasets with the following metrics:

* View confusion matrices in the `Images/` folder
* Model architecture and training details available in `scripts/face-emotion-detection.ipynb`

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is licensed under MIT terms specified in the LICENSE file.

---

## 🙏 Acknowledgments

* **Groq** for providing fast LLM inference
* **TMDB** for comprehensive movie data
* **TensorFlow/Keras** for deep learning framework
* **Streamlit** for the interactive web interface

---
