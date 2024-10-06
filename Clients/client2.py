from flask import Flask, render_template, request, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import requests
import io
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend suitable for server use
import matplotlib.pyplot as plt
from cryptography.fernet import Fernet
import json
import base64

app = Flask(__name__)
CORS(app)

key = b'A_5Mg2wyty1jnr4hxK3E7YvcuQSQBLV_qQqAKmkqwaM='
fernet = Fernet(key)

local_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(6,)),  # Input layer (6 features) + first hidden layer
        tf.keras.layers.Dense(32, activation='relu'),  # Second hidden layer
        tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])

def train_model(dataset_path):
    global local_model
    df = pd.read_csv(dataset_path)
    X, y = df[['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']].values, df['Output (S)'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    local_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = local_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    _, test_accuracy = local_model.evaluate(X_test, y_test)

    return history, f"Accuracy on Dataset: {test_accuracy*100:.2f}%"

def evaluate_model(dataset_path):
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
    
    weights_as_list = [w.tolist() for w in local_model.get_weights()]
    weights_json = json.dumps(weights_as_list)    
    encrypted_data = fernet.encrypt(weights_json.encode())
    encoded_data = base64.b64encode(encrypted_data).decode('utf-8')  # Convert to string    
    data = {'model_weights': encoded_data}
    
    response = requests.post(url, json=data)
    return response.text

def get_model_from_server():
    global local_weights
    url = 'http://127.0.0.1:5000/send_global_weights'
    response = requests.get(url)
    if response.status_code == 200:
        received_weights_encoded = response.json()['global_weights']   

        received_weights_bytes = base64.b64decode(received_weights_encoded)
        decrypted_weights_json = fernet.decrypt(received_weights_bytes).decode('utf-8')
        received_weights = json.loads(decrypted_weights_json)
        
        global_weights = [np.array(layer) for layer in received_weights]
        local_model.set_weights(global_weights)
        
        return "Model Weights changed to Global!"
    return "Error!"

def plot_training_graph(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Performance Over Epochs')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return img

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/execute', methods=['POST'])
def execute():
    global local_model
    result = None
    showplot = False
    data_path = "/Users/anand/Documents/College/Sem7/CryptoProject/Clients/client2_data.csv"
    if request.form['action'] == 'Train Model':
        history, result = train_model(data_path)
        showplot = True
    elif request.form['action'] == 'Send Model to Server':
        result = send_model_to_server()
    elif request.form['action'] == 'Get Global Weights':
        result = get_model_from_server()
    elif request.form['action'] == 'Evaluate Model':
        result = evaluate_model(data_path)
    return render_template('index.html', result=result, showplot=showplot)

@app.route('/plot.png')
def plot_png():
    data_path = "/Users/anand/Documents/College/Sem7/CryptoProject/Clients/client2_data.csv"
    history, _ = train_model(data_path)
    img = plot_training_graph(history)
    return send_file(img, mimetype='image/png')

if __name__ == "__main__":
    app.run(debug=True, port=5002)
