import os
import json
import cv2
import streamlit as st
import h5py
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator #type: ignore
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import Flatten, Dense #type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array #type: ignore
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input #type: ignore
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# Load pre-trained model (VGG16)
model = VGG16(include_top=False, input_shape=(224, 224, 3))

def extract_features(image_path):
    """Extract features using VGG16 model."""
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()

def load_features_from_hdf5(file="found_pet_features.h5"):
    """Load features from an HDF5 file."""
    with h5py.File(file, "r") as f:
        found_pets = list(f["file_paths"])
        found_pet_features = list(f["features"])

    # Decode bytes for file_paths and convert features to arrays
    found_pets = [pet.decode('utf-8') for pet in found_pets]  # Convert bytes to string
    found_pet_features = [np.array(features) for features in found_pet_features]

    return found_pets, found_pet_features

found_pets, found_pet_features = load_features_from_hdf5()

# Streamlit UI
st.title("Lost and Found Pet Matching")

# Upload lost pet image
uploaded_lost_image = st.file_uploader("Upload the image of the lost pet", type=["jpg", "jpeg", "png"])

# Button to trigger finding matching pets
if st.button("Find Matching Pets"):
    if uploaded_lost_image is not None:
        # Save uploaded image
        with open("lost_pet_image.jpg", "wb") as f:
            f.write(uploaded_lost_image.getbuffer())

        # Extract features from the uploaded lost pet image
        lost_pet_features = extract_features(uploaded_lost_image)

        # Calculate similarity between the lost pet and found pets
        similarities = [cosine_similarity([lost_pet_features], [features])[0][0] for features in found_pet_features]

        top_n = 3  # Number of top matches to retrieve
        sorted_indices = np.argsort(similarities)[::-1][:top_n]  # Get indices of top N similarities
        # Retrieve the top matches
        top_matches = [(found_pets[i], similarities[i]) for i in sorted_indices]
        
    #     # Display results
    #     st.image("lost_pet_image.jpg", caption="Lost Pet", use_column_width=True)
    #     st.write(f"Best match: {best_match_pet} (Similarity: {similarities[best_match_index]:.2f})")

    #     st.write(f"Found pets: {found_pets}")
    # else:
    #     st.error("Please upload a pet image to find a match.")
        
        
    
    # Step 4: Display the results with images in Streamlit
    if top_matches[0][1] > 0.5:  # Check if the top match's similarity is above the threshold
        st.write(f"Top {len(top_matches)} Matches:")
        
        for i, (pet_path, similarity) in enumerate(top_matches, start=1):
            # st.write(f"Rank {i}: {pet_path} (Similarity: {similarity:.2f})")
            st.image(pet_path, caption=f"Similarity: {similarity:.2f}", use_container_width=True)
    else:
        st.write(f"NO MATCHES FOUND. Please try again with a different image.")

