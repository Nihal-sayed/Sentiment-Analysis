from flask import Flask, render_template, request
import pickle
import re
import streamlit as st
from textblob import TextBlob

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Preprocess function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# Sentiment analysis function
def analyze_sentiment(text):
    cleaned_text = preprocess_text(text)
    analysis = TextBlob(cleaned_text)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review_text = request.form['review_text']
        sentiment = analyze_sentiment(review_text)
        return render_template('result.html', sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
