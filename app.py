import streamlit as st
import pandas as pd
import numpy as np
import os
import cv2
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the preprocessed dataset
@st.cache_data
def load_data():
    data = pd.read_csv("input/cleaned_data.csv") 
    return data

# Preprocessing pipeline
def preprocess_data(data, user_preferences):
    # Define features
    features = ['Type', 'Age(months)', 'Gender', 'MaturitySize', 'FurLength', 'FeeCategory', 
                'Health', 'Color1_Name']
    
    numeric_features = ['Age(months)']
    categorical_features = ['Type', 'Gender', 'MaturitySize', 'FurLength', 'Color1_Name', 'FeeCategory']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    # Transform dataset features
    X_preprocessed = preprocessor.fit_transform(data[features])

    # Transform user input
    user_input = pd.DataFrame([user_preferences])
    user_input_encoded = preprocessor.transform(user_input)

    return X_preprocessed, user_input_encoded

def is_image_clear(image_path, threshold=100):
    """
    Check if an image is clear based on the Laplacian variance method.

    Args:
        image_path (str): Path to the image file.
        threshold (float): Sharpness threshold. Lower values indicate blurry images.

    Returns:
        bool: True if the image is clear, False if it's blurry.
    """
    if not os.path.exists(image_path):
        return False

    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return False

    # Calculate the Laplacian variance
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
    return laplacian_var > threshold


def main():
    st.title("Pet Recommendation System")
    st.write("Get personalized pet recommendations based on your preferences!")

    # Load data
    data = load_data()

    # User input form
    st.sidebar.header("User Preferences")
    user_preferences = {
        'Type': st.sidebar.selectbox("Pet Type", data['Type'].unique()),
        'Age(months)': st.sidebar.slider("Age (in months)", 0, 255, 0),
        'Gender': st.sidebar.selectbox("Gender", data['Gender'].unique()),
        'MaturitySize': st.sidebar.selectbox("Maturity Size", data['MaturitySize'].unique()),
        'FurLength': st.sidebar.selectbox("Fur Length", data['FurLength'].unique()),
        'Color1_Name': st.sidebar.selectbox("Primary Color", data['Color1_Name'].unique()),
        'FeeCategory': st.sidebar.selectbox("Fee ranges", ['Low', 'Medium', 'High', 'Premium'])
    }

    # Preprocess data
    X_preprocessed, user_input_encoded = preprocess_data(data, user_preferences)

    # Compute similarity scores
    similarity_scores = cosine_similarity(user_input_encoded, X_preprocessed)[0]
    data['Similarity'] = similarity_scores

    # Filter top recommendations
    top_recommendations = data[(data['Gender'] == user_preferences['Gender']) & (data['Type'] == user_preferences['Type']) & (data['FeeCategory'] == user_preferences['FeeCategory'])].sort_values('Similarity', ascending=False)

    # Blur filtering and display recommendations
    st.subheader("Recommended Pets (Images)")
    displayed_count = 0
    max_recommendations = 5  # Display a maximum of 5 recommendations

    for _, pet in top_recommendations.iterrows():
        valid_images = []  # Collect clear images for this pet

        if isinstance(pet['image_paths'], str):
            image_paths = pet['image_paths'].strip("[]").replace("'", "").split(", ")
            
            for image_path in image_paths:
                abs_image_path = os.path.abspath(image_path)
                if os.path.exists(abs_image_path) and is_image_clear(abs_image_path, threshold=300):
                    valid_images.append(abs_image_path)
        
        # Skip if no valid images for this pet
        if not valid_images:
            continue

        # Display the pet's images
        valid_images = valid_images[:3]  # Limit to 3 images
        cols = st.columns(len(valid_images))
        for col, image_path in zip(cols, valid_images):
            with col:
                st.image(image_path, use_container_width=True)

        # Display pet details below images
        st.write(f"**Name:** {pet['Name']}")
        st.write(f"**Breed:** {pet['Breed1_Name']}")
        st.write(f"**Similarity Score:** {pet['Similarity']:.2f}")
        st.write(f"**Type:** {pet['Type']} | **Gender:** {pet['Gender']} | **Age:** {pet['Age(months)']} months")
        st.write(f"**Color:** {pet['Color1_Name']} | **Maturity Size:** {pet['MaturitySize']} | **Fur Length:** {pet['FurLength']}")
        st.write(f"**Fee:** ${pet['Fee']} | **Health:** {pet['Health']}")

        st.text("-" * 140)  # Separator

        # Increment displayed count and check if max reached
        displayed_count += 1
        if displayed_count >= max_recommendations:
            break

    # If no recommendations were displayed
    if displayed_count == 0:
        st.write("No suitable pets found with clear images. Try adjusting your preferences.")

if __name__ == "__main__":
    main()
