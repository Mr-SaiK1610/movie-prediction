import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Set page configuration
st.set_page_config(page_title="Movie Rating Predictor", layout="wide")

# Title and description
st.title("Movie Rating Prediction System")
st.write("Enter movie details to predict its rating")

# Create sample movie data
def create_sample_data():
    data = {
        'genre': ['Action', 'Comedy', 'Drama', 'Horror', 'Sci-Fi'] * 20,
        'budget': np.random.randint(1000000, 200000000, 100),
        'runtime': np.random.randint(60, 180, 100),
        'year': np.random.randint(1990, 2024, 100),
        'rating': np.random.uniform(1, 10, 100)
    }
    return pd.DataFrame(data)

# Load and prepare data
df = create_sample_data()
le = LabelEncoder()
df['genre_encoded'] = le.fit_transform(df['genre'])

# Train model
X = df[['genre_encoded', 'budget', 'runtime', 'year']]
y = df['rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Create input form
st.sidebar.header("Enter Movie Details")

genre = st.sidebar.selectbox("Select Genre", ['Action', 'Comedy', 'Drama', 'Horror', 'Sci-Fi'])
budget = st.sidebar.number_input("Enter Budget (USD)", min_value=1000000, max_value=500000000, value=50000000)
runtime = st.sidebar.number_input("Enter Runtime (minutes)", min_value=60, max_value=240, value=120)
year = st.sidebar.number_input("Enter Release Year", min_value=1990, max_value=2024, value=2023)

# Make prediction
if st.sidebar.button("Predict Rating"):
    # Prepare input data
    genre_encoded = le.transform([genre])[0]
    input_data = np.array([[genre_encoded, budget, runtime, year]])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    # Display prediction
    st.header("Prediction Results")
    st.write(f"Predicted Rating: {prediction:.1f}/10")
    
    # Display feature importance
    st.subheader("Feature Importance")
    importance_df = pd.DataFrame({
        'Feature': ['Genre', 'Budget', 'Runtime', 'Year'],
        'Importance': model.feature_importances_
    })
    st.bar_chart(importance_df.set_index('Feature'))

# Display sample data
st.subheader("Sample Training Data")
st.dataframe(df.head())