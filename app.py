import streamlit as st
import pickle
import re
from textblob import TextBlob

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

# Main Streamlit app
def main():
    st.title('Sentiment Analysis')

    review_text = st.text_area('Enter your review:')
    if st.button('Predict Sentiment'):
        sentiment = analyze_sentiment(review_text)
        st.write(f'The sentiment of the review is: {sentiment}')

if __name__ == '__main__':
    main()
