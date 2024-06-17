import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import linear_kernel
import tensorflow as tf
import numpy as np
import logging

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

model_path = '../models/models_category_location_user.h5'
model = tf.keras.models.load_model(model_path)
logging.info(f"Model loaded from {model_path}")

data_path = '../models/setelah_cleaning.csv'
data = pd.read_csv(data_path)
logging.info(f"Data loaded from {data_path}")

data['Category'] = data['Category'].fillna('')
data['Location_City'] = data['Location_City'].fillna('')

# Vectorize text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=10)
tfidf_matrix = tfidf_vectorizer.fit_transform(data['Category'])

# One-Hot Encode Location_City
onehot_encoder = OneHotEncoder(sparse_output=False)
location_matrix = onehot_encoder.fit_transform(data[['Location_City']])

# Combine TF-IDF matrix with Location City matrix
combined_matrix = np.hstack((tfidf_matrix.toarray(), location_matrix))

@app.route('/recommend', methods=['POST'])
def recommend_events():
    try:
        request_data = request.json
        logging.debug(f"Received request data: {request_data}")

        if 'Category' not in request_data or 'Location_City' not in request_data:
            return jsonify({'error': 'Category or Location_City key not found in JSON data'})

        input_categories = request_data['Category']
        input_location = request_data['Location_City']
        logging.debug(f"Input categories: {input_categories}")
        logging.debug(f"Input location: {input_location}")

        # Check if input_categories is a list
        if not isinstance(input_categories, list):
            return jsonify({'error': 'Category must be a list'})

        # Transform input categories to TF-IDF vectors and average them
        category_vecs = tfidf_vectorizer.transform(input_categories)
        category_vec = np.mean(category_vecs, axis=0).reshape(1, -1)

        # Transform input location to one-hot encoding
        location_vec = onehot_encoder.transform([[input_location]])
        logging.debug(f"Location vector shape: {location_vec.shape}")

        # Combine both vectors
        combined_input = np.hstack((category_vec, location_vec))
        logging.debug(f"Combined input shape: {combined_input.shape}")

        encoded_input = model.predict(combined_input)
        logging.debug(f"Encoded input shape: {encoded_input.shape}")

        # Calculate similarity
        cosine_similarities = linear_kernel(encoded_input, combined_matrix).flatten()
        logging.debug(f"Cosine similarities shape: {cosine_similarities.shape}")
        
        # Get the top recommendations
        related_docs_indices = cosine_similarities.argsort()[::-1]
        logging.debug(f"Related docs indices: {related_docs_indices}")

        recommended_events = []
        seen_titles = set()
        max_events = 20

        # Prioritize events with matching category and location
        for idx in related_docs_indices:
            event = data.iloc[idx]
            if event['Category'] in input_categories and event['Location_City'] == input_location:
                if len(recommended_events) < max_events:
                    recommended_events.append({
                        'Title': event['Title'],
                        'Category': event['Category'],
                        'Event_Start': event['Event_Start'],
                        'Location_City': event['Location_City'],
                        'Location_Address': event['Location_Address']
                    })
                    seen_titles.add(event['Title'])

        # Add events with matching category but different location
        for idx in related_docs_indices:
            event = data.iloc[idx]
            if event['Category'] in input_categories and event['Location_City'] != input_location:
                if len(recommended_events) < max_events:
                    recommended_events.append({
                        'Title': event['Title'],
                        'Category': event['Category'],
                        'Event_Start': event['Event_Start'],
                        'Location_City': event['Location_City'],
                        'Location_Address': event['Location_Address']
                    })
                    seen_titles.add(event['Title'])

        # Add a limited number of events with different category
        additional_events = []
        for idx in related_docs_indices:
            event = data.iloc[idx]
            if event['Category'] not in input_categories and event['Title'] not in seen_titles:
                additional_events.append({
                    'Title': event['Title'],
                    'Category': event['Category'],
                    'Event_Start': event['Event_Start'],
                    'Location_City': event['Location_City'],
                    'Location_Address': event['Location_Address']
                })
                seen_titles.add(event['Title'])
            if len(additional_events) >= 5:
                break

        recommended_events.extend(additional_events[:5])

        logging.debug(f"Final selected events: {recommended_events}")

        return jsonify({'recommended_events': recommended_events})

    except Exception as e:
        logging.error(f"Error in /recommend endpoint: {str(e)}")
        return jsonify({'error': 'An error occurred. Please try again later.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
