import pandas as pd
import joblib
import json

def predict_car_price(input_data):
    # Load the trained model
    model = joblib.load('car_price_model.pkl')

    # Load feature columns saved during training
    with open('feature_columns.json', 'r') as f:
        feature_columns = json.load(f)

    # Convert input dictionary to DataFrame
    input_df = pd.DataFrame([input_data])

    # One-hot encode categorical features
    input_df = pd.get_dummies(input_df)

    # Reindex to match training features; fill missing columns with 0
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    # Predict the price
    predicted_price = model.predict(input_df)[0]
    return predicted_price

if __name__ == "__main__":
    # Example input dictionary with sample values
    sample_input = {
        'symboling': 3,
        'fueltype': 'gas',
        'aspiration': 'std',
        'doornumber': 'four',
        'carbody': 'sedan',
        'drivewheel': 'fwd',
        'enginelocation': 'front',
        'wheelbase': 88.6,
        'carlength': 168.8,
        'carwidth': 64.1,
        'carheight': 48.8,
        'curbweight': 2548,
        'enginetype': 'ohc',
        'cylindernumber': 'four',
        'enginesize': 130,
        'fuelsystem': '2bbl',
        'boreratio': 3.47,
        'stroke': 2.68,
        'compressionratio': 9.0,
        'horsepower': 111,
        'peakrpm': 5000,
        'citympg': 21,
        'highwaympg': 27
    }

    price = predict_car_price(sample_input)
    print(f"Predicted car price: {price:.2f}")
