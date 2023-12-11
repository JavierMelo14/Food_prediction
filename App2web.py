from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

model_Food = joblib.load('food_prediction.joblib')

df = pd.read_csv('df_final.csv')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    city = request.form.get('city')
    print("City from form:", city)
    
    category = request.form.get('category')
    print(category)
    
    unit = request.form.get('unit') 
    print(unit)

    input_data = df[['City', 'category', 'unit']]

    input_data.loc[0] = [city, category, unit]
    print(input_data.head(5))

    all_data_encoded = pd.get_dummies(df, columns=['City', 'category', 'unit'], drop_first=True)

    input_data_encoded = all_data_encoded[all_data_encoded['City_' + str(city)] == 1]

    print("input_data_encoded:", input_data_encoded)

    prediction = model_Food.predict(input_data_encoded)
    print("prediction:", prediction)

    prediction = prediction[0] if isinstance(prediction, (list, np.ndarray)) else prediction

    return render_template('result.html', city=city, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)