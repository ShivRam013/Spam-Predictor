# 📩 SMS Spam Detection System

A machine learning project that classifies SMS messages as **Spam** or **Ham (Not Spam)** using Natural Language Processing (NLP) and Naive Bayes classification.

---

## 🚀 Overview

This project builds an end-to-end text classification pipeline on the [Kaggle SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset). It preprocesses raw SMS text, extracts features using TF-IDF, and compares multiple Naive Bayes models to find the best performer.

---

## 📊 Model Performance

| Model | Accuracy | Precision |
|-------|----------|-----------|
| Gaussian NB | 86.4% | 50.8% |
| Multinomial NB | 96.1% | 99.1% |
| **Bernoulli NB** | **96.6%** | **96.6%** |

✅ **Bernoulli Naive Bayes** selected as the final model based on overall accuracy.

---

## 🛠️ Tech Stack

- **Language:** Python
- **NLP:** NLTK (tokenization, stopword removal, stemming)
- **Feature Extraction:** TF-IDF Vectorization (Scikit-Learn)
- **Models:** Scikit-Learn (GaussianNB, MultinomialNB, BernoulliNB)
- **EDA:** Matplotlib, Seaborn, WordCloud
- **Data Handling:** Pandas, NumPy

---

## 🔄 Pipeline

```
Raw SMS Text
    ↓
Lowercasing → Tokenization → Stopword Removal → Punctuation Removal → Stemming
    ↓
TF-IDF Vectorization
    ↓
Naive Bayes Model (Bernoulli NB)
    ↓
Spam / Not Spam
```

---

## 📁 Project Structure

```
Spam-Predictor/
│
├── spam.csv                  # Dataset (Kaggle)
├── spam_py_code_.ipynb       # Main notebook
├── spam_model.pkl            # Saved model (Pickle)
├── vector.pkl                # Saved TF-IDF vectorizer
└── README.md
```

---

## ▶️ How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/ShivRam013/Spam-Predictor.git
   cd Spam-Predictor
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy scikit-learn nltk matplotlib seaborn wordcloud
   ```

3. **Run the notebook**
   ```bash
   jupyter notebook spam_py_code_.ipynb
   ```

4. **Test a message**
   ```
   Enter your SMS: Congratulations! You've won a free prize. Call now!
   Output: Spam Detected
   ```

---

## 📌 Dataset

- Source: [Kaggle - SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- 5,572 SMS messages labeled as spam or ham
- After removing duplicates: ~5,169 messages

---

## 👤 Author

**Shiv Ram Mahto**
BCA Student | GGSIPU, New Delhi
[GitHub](https://github.com/ShivRam013)
