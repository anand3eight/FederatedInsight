from flask import Flask, render_template, request
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)

# Placeholder for client model weights
client_model_weights = []
global_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(6,)),  # Input layer
        tf.keras.layers.Dense(32, activation='relu'),  # Hidden layer
        tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer
    ])

def aggregate_models():
    global client_model_weights, global_model
    # Aggregate by averaging weights across clients
    if len(client_model_weights) == 0 :
        return "Currently Not Possible!"
    average_weights = [np.mean([weights[layer] for weights in client_model_weights], axis=0)
                       for layer in range(len(client_model_weights[0]))]            
    global_model.set_weights(average_weights)
    return "Simple average aggregation completed!"

# Evaluate the model
def evaluate_aggregated_model(dataset_path):
    global average_weights
    if len(client_model_weights) == 0 :
        return "Currently Not Possible!"

    df = pd.read_csv(dataset_path)
    X, y = df[['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']].values, df['Output (S)'].values
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_test = scaler.fit_transform(X_test)

    global_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    _, test_accuracy = global_model.evaluate(X_test, y_test)
    
    return f"Aggregated Model Accuracy: {test_accuracy * 100:.2f}%"

@app.route('/')
def home():
    return render_template('server_index.html')

@app.route('/aggregate', methods=['POST'])
def aggregate():
    global client_model_weights
    result = aggregate_models()
    return render_template('server_index.html', result=result)

@app.route('/evaluate', methods=['POST'])
def evaluate():
    dataset_path = "/Users/anand/Documents/College/Sem7/CryptoProject/Server/detect_dataset.csv"
    result = evaluate_aggregated_model(dataset_path)
    return render_template('server_index.html', result=result)

@app.route('/receive_model', methods=['POST'])
def receive_model():
    global client_model_weights
    received_weights = request.json['model_weights']
    client_model_weights.append(received_weights)
    return "Model weights received from client."

@app.route('/send_global_weights', methods=['GET'])
def send_global_weights():
    global global_model 
    weights_as_list = [w.tolist() for w in global_model.get_weights()]
    return {"global_weights" : weights_as_list}

if __name__ == "__main__":
    app.run(debug=True, port=5000)
