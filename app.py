import os
import shutil
import re
import string
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
import streamlit as st
import time
# import warnings

# Suppress TensorFlow warnings
# tf.get_logger().setLevel('ERROR')

# # Suppress other warnings
# warnings.filterwarnings('ignore')

# Define constants
URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
DATASET_DIR = 'aclImdb'
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
BATCH_SIZE = 32
SEED = 42
MAX_FEATURES = 10000
SEQUENCE_LENGTH = 250
EMBEDDING_DIM = 16
MODEL_PATH = 'sentiment_model.keras'

# Function to download and extract dataset
def download_and_prepare_data():
    if not os.path.exists(DATASET_DIR):
        dataset = tf.keras.utils.get_file("aclImdb_v1", URL,
                                            untar=True, cache_dir='.',
                                            cache_subdir='')
        dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
        
        remove_dir = os.path.join(dataset_dir, 'train', 'unsup')
        if os.path.exists(remove_dir):
            shutil.rmtree(remove_dir)
    return os.path.join(DATASET_DIR, 'train')

# Load datasets
def load_datasets():
    raw_train_ds = tf.keras.utils.text_dataset_from_directory(
        TRAIN_DIR,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        subset='training',
        seed=SEED)

    raw_val_ds = tf.keras.utils.text_dataset_from_directory(
        TRAIN_DIR,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        subset='validation',
        seed=SEED)

    raw_test_ds = tf.keras.utils.text_dataset_from_directory(
        os.path.join(DATASET_DIR, 'test'),
        batch_size=BATCH_SIZE)

    return raw_train_ds, raw_val_ds, raw_test_ds

# Define custom standardization
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation),
                                    '')

# Create and adapt vectorization layer
def create_and_adapt_vectorizer(train_ds):
    vectorize_layer = layers.TextVectorization(
        standardize=custom_standardization,
        max_tokens=MAX_FEATURES,
        output_mode='int',
        output_sequence_length=SEQUENCE_LENGTH)
    
    train_text = train_ds.map(lambda x, y: x)
    vectorize_layer.adapt(train_text)
    
    return vectorize_layer

# Vectorize text
def vectorize_text(text, vectorize_layer):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text)

# Optimize dataset performance
AUTOTUNE = tf.data.AUTOTUNE
def optimize_datasets(train_ds, val_ds, test_ds):
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds, test_ds

# Define the model
def create_model():
    model = tf.keras.Sequential([
        layers.Embedding(MAX_FEATURES, EMBEDDING_DIM),
        layers.Dropout(0.2),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss=losses.BinaryCrossentropy(),
                  optimizer='adam',
                  metrics=[tf.metrics.BinaryAccuracy(threshold=0.5)])
    return model

# Train the model if it does not exist
def train_model():
    raw_train_ds, raw_val_ds, _ = load_datasets()
    vectorize_layer = create_and_adapt_vectorizer(raw_train_ds)
    
    train_ds = raw_train_ds.map(lambda x, y: vectorize_text(x, vectorize_layer))
    val_ds = raw_val_ds.map(lambda x, y: vectorize_text(x, vectorize_layer))
    train_ds, val_ds, _ = optimize_datasets(train_ds, val_ds, None)
    
    model = create_model()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5,  # Adjust as needed
        verbose=1)
    
    model.save(MODEL_PATH)  # Save the entire model
    return model, history

# Load or train model
@st.cache_resource
def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
    else:
        model, _ = train_model()
    return model

# Load model and vectorizer
model = load_or_train_model()
raw_train_ds, _, _ = load_datasets()
vectorize_layer = create_and_adapt_vectorizer(raw_train_ds)

# Prediction function
def predict_sentiment(text):
    vectorized_text = vectorize_text(tf.constant([text]), vectorize_layer)
    prediction = model.predict(vectorized_text)
    sentiment = 'Positive' if prediction >= 0.5 else 'Negative'
    return sentiment

# Streamlit application
st.title("Sentiment Analysis Web App")
st.write("Enter a movie review below and get the sentiment analysis result:")

user_input = st.text_area("Review Text")

if st.button("Analyze"):
    if user_input:
        with st.spinner('Analyzing sentiment...'):
            time.sleep(1)  
            sentiment = predict_sentiment(user_input)
            st.success(f"Sentiment: {sentiment}")
    else:
        st.error("Please enter some text.")
