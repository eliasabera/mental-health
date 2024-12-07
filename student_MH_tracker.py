import pandas as pd
import datetime as dt
import streamlit as st
import plotly.graph_objects as go
import base64

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
import streamlit as st

# Custom CSS for navigation
with open('css.css') as f:
    css = f.read()
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# Home Page
def home_page():
    st.title("Welcome to Mental Health Tracker")
    st.write("Your journey to mental well-being starts here.")
    st.image("mind.jpg", use_column_width=True)
    st.write("Navigate to different sections using the links below.")
    # Navigation Buttons


col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Go to Well-being Tracker"):
        st.session_state.current_page = "Well-being"

with col2:
    if st.button("Go to Academic Tracker"):
        st.session_state.current_page = "Academic Tracker"

with col3:
    if st.button("Go to Journal"):
        st.session_state.current_page = "Journal"

# Well-being Page
def wellbeing_page():
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

# Academic Tracker Page
def academic_tracker_page():
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
# Journal Page
def journal_page():
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

# Main App Logic
def main():
    # Initialize the current page
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Home"

    # Page Navigation Logic
    if st.session_state.current_page == "Home":
        home_page()
    elif st.session_state.current_page == "Well-being":
        wellbeing_page()
    elif st.session_state.current_page == "Academic Tracker":
        academic_tracker_page()
    elif st.session_state.current_page == "Journal":
        journal_page()

# Run the app
if __name__ == "__main__":
    main()
