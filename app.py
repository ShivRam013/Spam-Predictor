import streamlit as st
import pandas as pd
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def trans_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text))

    y = []
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation and i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)


tfidf = pickle.load(open('vector.pkl', 'rb'))
model = pickle.load(open('spam_model.pkl', 'rb'))

st.title('Spam Classifier')
input_sms = st.text_area("Enter message")

if st.button('Predict'):


    trans_sms = trans_text(input_sms)
    vector_input = tfidf.transform([trans_sms])
    result = model.predict(vector_input)[0]

    if result == 1:
        st.text('Spam Detected')
    else:
        st.text('Not Spam')