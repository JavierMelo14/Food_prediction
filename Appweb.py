from flask import Flask, render_template, request
import pandas as pd
from joblib import load

app = Flask(__name__)
model = load('food_prediction.joblib')

def preprocess_input(city):
    # Realiza un one-hot encoding simple para la ciudad
    df_input = pd.DataFrame({'City': [city]})
    X_input = pd.get_dummies(df_input, columns=['City'], drop_first=True)

    # Asegúrate de que las columnas estén en el mismo orden que durante el entrenamiento
    expected_columns = set(model.feature_importances_)
    input_columns = set(X_input.columns)
    missing_columns = expected_columns - input_columns

    if missing_columns:
        # Agrega columnas faltantes con valores 0
        for column in missing_columns:
            X_input[column] = 0

    # Reordena las columnas para que coincidan con el orden durante el entrenamiento
    X_input = X_input[sorted(X_input.columns)]

    return X_input.values.flatten()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Captura la ciudad desde el formulario
    city = request.form['city']

    # Realiza validaciones y preprocesamiento
    validation_result, features = validate_and_preprocess(city)
    if not validation_result:
        return render_template('error.html', message='Invalid input')

    # Realiza la predicción con el modelo
    prediction = make_prediction(features)

    # Muestra el resultado al usuario
    return render_template('result.html', city=city, prediction=prediction)

def validate_and_preprocess(city):
    # Ejemplo simple de validación y preprocesamiento
    # Puedes personalizar esta función según tus necesidades

    # Validar si la ciudad no es nula o vacía
    if not city:
        return False, None

    # Aplicar preprocesamiento de entrada
    features = preprocess_input(city)

    return True, features

def make_prediction(features):
    # Utiliza el modelo para realizar la predicción
    # Asegúrate de que la entrada esté en el formato correcto para tu modelo
    prediction = model.predict([features])

    return prediction[0]

if __name__ == '__main__':
    app.run(debug=True)





