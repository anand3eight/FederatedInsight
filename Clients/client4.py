from flask import Flask, render_template, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import requests

app = Flask(__name__)
CORS(app)

local_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(6,)),  # Input layer (6 features) + first hidden layer
        tf.keras.layers.Dense(32, activation='relu'),  # Second hidden layer
        tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])

def train_model(dataset_path) :
    global local_model
    df = pd.read_csv(dataset_path)
    X, y = df[['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']].values, df['Output (S)'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    local_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    local_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    _, test_accuracy = local_model.evaluate(X_test, y_test)
    return f"Accuracy on Dataset : {test_accuracy*100:.2f}%"

def evaluate_model(dataset_path) :
    global local_model
    df = pd.read_csv(dataset_path)
    X, y = df[['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']].values, df['Output (S)'].values
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_test = scaler.fit_transform(X_test)

    local_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    _, test_accuracy = local_model.evaluate(X_test, y_test)
    return f"Aggregated Model Accuracy: {test_accuracy * 100:.2f}%"

def send_model_to_server():
    global local_model
    url = 'http://127.0.0.1:5000/receive_model'  # The server endpoint
    weights_as_list = [ w.tolist() for w in local_model.get_weights() ]
    data = {'model_weights': weights_as_list}  # Prepare the weights for sending
    response = requests.post(url, json=data)
    return response.text


def get_model_from_server():
    global local_weights

    url = 'http://127.0.0.1:5000/send_global_weights'     
    response = requests.get(url)
    if response.status_code == 200 :
        global_weights = [np.array(layer) for layer in response.json()['global_weights']]
        local_model.set_weights(global_weights)  
        return "Model Weights changed to Global!"
    return "Error!"


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/execute', methods=['POST'])
def execute():
    global local_model
    result = None
    data_path = "/Users/anand/Documents/College/Sem7/CryptoProject/Clients/client4_data.csv"
    if request.form['action'] == 'Train Model':
        result = train_model(data_path)
    elif request.form['action'] == 'Send Model to Server':
        result = send_model_to_server()
    elif request.form['action'] == 'Get Global Weights':
        result = get_model_from_server()
    elif request.form['action'] == 'Evaluate Model':
        result = evaluate_model(data_path)
    return render_template('index.html', result=result)

if __name__ == "__main__" :
   app.run(debug=True, port=5004)
    
    