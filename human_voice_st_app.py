import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

 
# Load trained model and scaler
model = joblib.load(r"c:\Users\Lenovo\Desktop\GUVI\PROJECT HUMAN VOICE CLASSIFICATION\model_13.pkl")
scaler = joblib.load(r"c:\Users\Lenovo\Desktop\GUVI\PROJECT HUMAN VOICE CLASSIFICATION\scaler_13.pkl")

# Top 13 most important features with typical value ranges
top_13_features_info = {
    'mfcc_4_std': (12.0, 59.0),
    'mfcc_5_std': (4.0, 44.0),
    'mfcc_4_mean': (-7.0, 92),
    'mfcc_5_mean': (-50.0, 22.0),
    'mean_spectral_flatness': (0.0017, 0.0721),
    'mfcc_1_mean': (-448, -162),
    'mfcc_7_mean': (-34, 18),
    'mfcc_1_std': (52, 206),
    'mfcc_8_mean': (-34, 18),
    'zero_crossing_rate': (0.02, 0.27),
    'mfcc_2_std': (19, 109),
    'mfcc_10_mean': (-20, 19),
    'mfcc_12_mean': (-13, 16),
}

def generate_sample_data(num_samples=100):
    data = {
        feature: np.random.uniform(low, high, num_samples)
        for feature, (low, high) in top_13_features_info.items()
    }
    return pd.DataFrame(data)

# App Navigation
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ["Intro", "Visualization", "Feature Selection","Predict Gender"])

# Section 1: Introduction
if selection == "Intro":
    st.title("Top 13 Audio Features Analysis")
    st.markdown("""
    This Streamlit app explores the 13 most important audio features for classification tasks to predict gender that it is male or female.
    
    - Features are derived from MFCC and spectral properties.
    - You can explore the data, visualize distributions, and check the normalized feature ranges.
    
    **Sections**:
    - 📘 Intro
    - 📊 Visualization
    - 🧠 Feature Selection
    - 🔮 Predict gender
    """)

# Section 2: Visualization
elif selection == "Visualization":
    st.title("📊 Visualization of Top 13 Features")

    df = generate_sample_data()

    selected_feature = st.selectbox("Select a feature to visualize", df.columns)
    st.write(f"Distribution of `{selected_feature}`")

    fig, ax = plt.subplots()
    ax.hist(df[selected_feature], bins=30, color='skyblue', edgecolor='black')
    ax.set_xlabel(selected_feature)
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    st.write("Summary Statistics:")
    st.dataframe(df[selected_feature].describe().to_frame())

#  Section 3: Feature Selection
elif selection == "Feature Selection":
    st.title("🧠 Feature Range Overview")
    st.markdown("Below are the top 13 selected features with their min and max value ranges.")
    feature_df = pd.DataFrame(top_13_features_info).T
    feature_df.columns = ['Min', 'Max']
    st.dataframe(feature_df)

# Section 4: Predict Voice
elif selection == "Predict Gender":
    st.title("🔮 Predict Voice Type")
    st.markdown("Use the sliders to input values for each feature, then click **Predict gender**.")

    input_values = []
    for feature, (min_val, max_val) in top_13_features_info.items():
        if max_val - min_val > 1.0:
            value = st.slider(f"{feature}", min_value=float(min_val), max_value=float(max_val), step=0.01)
        else:
            value = st.slider(f"{feature}", min_value=float(min_val), max_value=float(max_val), step=0.01)
        input_values.append(value)
    # Prediction
    if st.button("Predict Gender"):
       try:
          input_array = np.array(input_values).reshape(1, -1)
          input_scaled = scaler.transform(input_array)
          pred = model.predict(input_scaled)[0]
          pred_prob = model.predict_proba(input_scaled)[0][pred] if hasattr(model, "predict_proba") else None

          gender = "Male" if pred == 1 else "Female"
          st.success(f"🎯 Predicted Gender: **{gender}**")
       except Exception as e:
           st.error(f"Prediction failed: {e}") 
       