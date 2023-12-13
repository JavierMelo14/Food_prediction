from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

model_Food = joblib.load('food_prediction.joblib')
model_hotel = joblib.load('hotel_prediction.joblib')

df = pd.read_csv('df_final.csv')
dfh = pd.read_csv('df_final_hotel.csv')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
        
    city = request.form.get('city')        
    category = request.form.get('category')        
    unit = request.form.get('unit')
        
    input_data = df[['City', 'category', 'unit']]
    input_data.loc[0] = [city, category, unit]

    input_data_encoded = pd.get_dummies(input_data, columns=['City', 'category', 'unit'], drop_first=True)
    input_data_encoded = input_data_encoded[input_data_encoded['City_' + str(city)] == 1]

    prediction = model_Food.predict(input_data_encoded)
    
    prediction = prediction[0] if isinstance(prediction, (list, np.ndarray)) else prediction

    return render_template('result.html', city=city, prediction=prediction)

@app.route('/predicthotel', methods=['POST'])
def predicthotel():
    rating = request.form.get('rating')
    rating_description = request.form.get('rating_description')
    reviews = request.form.get('reviews')
    star_rating = request.form.get('star_rating')
    location = request.form.get('location')
    nearest_landmark = request.form.get('nearest_landmark')
    distance_to_landmark = request.form.get('distance_to_landmark')
    tax = request.form.get('tax')

    input_data_hotel = dfh[['Rating', 'Rating Description', 'Reviews', 'Star Rating', 'Location', 'Nearest Landmark', 'Distance to Landmark', 'Tax']]
    input_data_hotel.loc[0] = [float(rating), str(rating_description), int(reviews), float(star_rating), str(location), str(nearest_landmark), float(distance_to_landmark), float(tax)]

    input_data_encoded_hotel= pd.get_dummies(input_data_hotel, columns=['Rating Description', 'Location', 'Nearest Landmark'], drop_first=True)
    input_data_encoded_hotel = input_data_encoded_hotel[input_data_encoded_hotel['Rating Description_' + str(rating_description)] == 1]

    prediction_hotel = model_hotel.predict(input_data_encoded_hotel)
    prediction_hotel = prediction_hotel[0] if isinstance(prediction_hotel, (list, np.ndarray)) else prediction_hotel

    return render_template('results_hotel.html', prediction_hotel=prediction_hotel)
if __name__ == '__main__':
    app.run(debug=True)