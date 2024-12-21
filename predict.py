import streamlit as st
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
# Load the trained model
MODEL_PATH = 'lnPredictor.pkl'  # Replace with your model file path
with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

# Streamlit app
def main():
    st.title("House Price Prediction")
    st.write("This app predicts the price of a house based on its features.")
    with open("predictCol.pkl", "rb") as file:
        xCols = pickle.load(file)
    with open("LabelEncoder.pkl", "rb") as file:
        encoders = pickle.load(file)
    xTest = []
    for col in xCols:
        if col in encoders:
            inputStr = st.selectbox(col, encoders[col].classes_)
            encodedVal = encoders[col].transform([inputStr])[0]
            xTest.append(encodedVal)
        else:
            xTest.append(st.number_input(col, min_value=0, step=1))
    
    

    # Collect features into a numpy array
    xTest = np.array(xTest)
    xTest = xTest.reshape(1, -1)
    # Predict button
    if st.button("Predict Price"):
        prediction = model.predict(xTest)
        st.success(f"The predicted price of the house is: {prediction[0]:,.2f} billion dong")

if __name__ == "__main__":
    main()
