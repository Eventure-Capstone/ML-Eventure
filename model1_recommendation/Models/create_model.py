import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import joblib

# Load datasets
# data = pd.read_csv('/content/drive/MyDrive/data-rekomendasi/user-lat-long.csv')
data = pd.read_csv('/content/user-lat-long.csv')

# Process data
data['Category'] = data['Category'].fillna('')

# Vectorize text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=10)
tfidf_matrix = tfidf_vectorizer.fit_transform(data['Category'])

# Convert TF-IDF matrix to dense NumPy array
tfidf_matrix_dense = tfidf_matrix.toarray()

# Encode Location_City using OneHotEncoder
location_encoder = OneHotEncoder()
location_encoded = location_encoder.fit_transform(data[['Location_City']]).toarray()

# Combine TF-IDF features with Location_City features
combined_features = np.hstack([tfidf_matrix_dense, location_encoded])

# Calculate cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

def train_model():
    # Define the recommendation model
    input_layer = Input(shape=(combined_features.shape[1],), name='input')
    encoded = Dense(100, activation='relu')(input_layer)
    decoded = Dense(combined_features.shape[1], activation='softmax')(encoded)

    # Autoencoder model
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    autoencoder.fit(combined_features, combined_features, epochs=10, batch_size=32)

    return autoencoder

if __name__ == '__main__':
    model = train_model()
    model.save('models_lat_long_city.h5')

    
    