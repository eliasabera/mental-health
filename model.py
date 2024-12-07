import pandas as pd
import numpy as np
import re
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from tensorflow import keras
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab',quiet=True)
nltk.download('wordnet', quiet=True)

# Setting constants
stopwords_list = set(stopwords.words('english'))
wnl = WordNetLemmatizer()

# Initialize the Tokenizer
word_tokenizer = Tokenizer()

# Function to preprocess text (remove special characters, stopwords, tokenize, lemmatize)
def preprocess_text(journal_text):
    # Remove special characters and digits
    journal_text = re.sub(r"[^a-zA-Z]", " ", journal_text)

    # Convert text to lower case
    journal_text = journal_text.lower()

    # Remove stopwords
    pattern = re.compile(r'\b(' + r'|'.join(stopwords_list) + r')\b\s*')
    text = pattern.sub('', journal_text)

    # Tokenization
    tokens = word_tokenize(text)

    # Lemmatization
    lemmatized_text = ' '.join([wnl.lemmatize(t) for t in tokens])

    return lemmatized_text

# Function to train the model and save it
def train_and_save_model():
    # Read data from CSV (replace with the correct file path)
    data = pd.read_csv('journal_data.csv')  # Ensure this file exists in the correct path

    # Check the columns in the data (ensure the column name for journal text is correct)
    print(data.columns)

    # Assuming the column with text is named 'journal_text'
    data['clean_text'] = data['journal_text'].apply(preprocess_text)

    # Tokenizing the text data
    word_tokenizer.fit_on_texts(data['clean_text'])
    X = word_tokenizer.texts_to_sequences(data['clean_text'])

    # Padding sequences to ensure uniform input length
    X = pad_sequences(X, maxlen=300, dtype='int32', padding='post', truncating='post', value=0.0)

    # Assuming 'mood' is the column with the label (0 for negative, 1 for positive)
    y = data['mood'].values

    # Building the model (BiLSTM model)
    model = keras.Sequential([
        keras.layers.Embedding(input_dim=len(word_tokenizer.word_index) + 1, output_dim=100, input_length=300),
        keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)),
        keras.layers.GlobalMaxPooling1D(),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')  # Binary classification (stressed: 1, not stressed: 0)
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X, y, epochs=5, batch_size=32, validation_split=0.2)

    # Save the trained model
    model.save('BiLSTM_model.h5')  # Save as .h5 for better compatibility

# Run the training and save the model
train_and_save_model()
