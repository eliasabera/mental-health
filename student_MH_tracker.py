import pandas as pd
import datetime as dt
import streamlit as st
import plotly.graph_objects as go

from PIL import Image
import joblib
import numpy as np
import nltk
import keras
import tensorflow as tf
import re
import pickle


from sqlalchemy.testing.plugin.plugin_base import warnings

import tracker_db
from tracker_db import *

# Downloading required NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab',quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize NLTK components
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
#from tensorflow.keras.preprocessing.text import Tokenizer

# Setting constants for stopwords
stopwords_list = set(stopwords.words('english'))

# Load the pre-trained sentiment model
sentiment_model = keras.models.load_model('BiLSTM_model.h5')


# Preprocess the journal text (cleaning and tokenizing)
def preprocess(journal_text):
    journal_text = re.sub(r"[^a-zA-Z]", " ", journal_text)
    journal_text = journal_text.lower()
    pattern = re.compile(r'\b(' + r'|'.join(stopwords_list) + r')\b\s*')
    text = pattern.sub('', journal_text)
    tokens = word_tokenize(text)
    wnl = WordNetLemmatizer()
    lemmatized_text = ' '.join([wnl.lemmatize(t) for t in tokens])
    text_list = [lemmatized_text]
    word_tokenizer = Tokenizer()
    word_tokenizer.fit_on_texts(text_list)
    text_sequence = word_tokenizer.texts_to_sequences(text_list)
    text_sequence = tf.keras.utils.pad_sequences(text_sequence, maxlen=300, dtype='int32', padding='post',
                                                 truncating='post', value=0.0)
    return text_sequence


# Function to predict journal sentiment (mood)
def journal_prediction(clean_journal_text):
    sentiment = sentiment_model.predict(clean_journal_text)
    return 1 if sentiment >= 0.5 else 0


# Custom CSS to make the app more visually appealing
with open('css.css') as f:
    css = f.read()

st.markdown(f'<style>{css} </style>', unsafe_allow_html=True)

with open('app.js') as f:
    app = f.read()

st.markdown(f'<script>{app} </script>', unsafe_allow_html=True)
# Main application
def main():
    st.title(" Mental Health Tracker ")
    st.sidebar.header("🌟 Menu 🌟")

    # Menu navigation options
    menu = ["Home", "Well-being", "Academic Tracker", "Journal"]
    choice = st.sidebar.selectbox("Choose a Section", menu)


    if choice == "Home":

        st.write("""" Breathe. Reflect. Grow: Your Journey to Mental Wellness.""")

    elif choice == "Well-being":
        st.header("💫 Track Your Well-being 💫")
        st.write("""
            It's important to regularly monitor your mood, sleep hours, and coping strategies to maintain a good mental state.
        """)

        with st.form("Well-being Form"):
            answer_list1, answer_list2 = get_user_input()
            submit = st.form_submit_button("Submit Well-being Data")

            if submit:
                try:
                    create_table(answer_list1, answer_list2)
                    st.success("Your well-being data has been logged successfully!")
                except Exception as e:
                    st.error(f"Error: {e}")

    elif choice == "Academic Tracker":
        st.header("📚 Track Your Academic Progress 📚")
        st.write("""
            Keep track of your academic workload, productivity, and deadlines to stay organized and reduce stress.
        """)

        with st.form("Academic Tracker Form"):
            answer_list1, answer_list2 = get_user_input()
            submit = st.form_submit_button("Submit Academic Data")

            if submit:
                try:
                    create_table(answer_list1, answer_list2)
                    st.success("Academic data logged successfully!")
                except Exception as e:
                    st.error(f"Error: {e}")

    elif choice == "Journal":
        st.header("📝 Journal Your Thoughts 📝")
        st.write("""
            Writing down your thoughts can help you reflect on your day and track your emotional state over time.
            This journal section will allow you to record your feelings and moods.
        """)

        with st.form("Journal Form"):
            st.markdown(
                "<div style='background-color: red;'>",
                unsafe_allow_html=True
            )
            journal_entry = st.text_area("Write your journal entry here:", "")
            entry_score = st.slider("Rate your sentiment (0-100):", 0, 100, 50)
            submit = st.form_submit_button("Submit Journal Entry")

            st.markdown("</div>", unsafe_allow_html=True)

            if submit:
                try:
                    now = dt.datetime.now()
                    date_string = now.strftime('%Y-%m-%d')

                    # Save the journal entry to the database
                    create_table([None, None, "None", 0, journal_entry, entry_score], [0, 0, 0, 0])
                    st.success("Your journal entry has been saved successfully!")
                except Exception as e:
                    st.error(f"Error: {e}")

    # Button to visualize data



# Run the app
if __name__ == "__main__":
    main()
