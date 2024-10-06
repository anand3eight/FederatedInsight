from flask import Flask, render_template, request
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from cryptography.fernet import Fernet
import json
import base64

app = Flask(__name__)
CORS(app)

key = b'A_5Mg2wyty1jnr4hxK3E7YvcuQSQBLV_qQqAKmkqwaM='
fernet = Fernet(key)

# Placeholder for client model weights
client_model_weights = []
global_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(6,)),  # Input layer
        tf.keras.layers.Dense(32, activation='relu'),  # Hidden layer
        tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer
    ])

def aggregate_models():
    global client_model_weights, global_model
    
    # Check if there are client weights to aggregate
    if len(client_model_weights) == 0:
        return "Currently Not Possible!"
    
    # Weights for weighted averaging (based on each client's data size)
    client_data_sizes = [3000, 3000, 3000, 3000]  # Example sizes for each client
    total_data_size = sum(client_data_sizes)
    normalized_weights = [size / total_data_size for size in client_data_sizes]
    
    # Aggregating the weights manually
    aggregated_weights = []
    
    # Iterate through layers and aggregate
    for layer in range(len(client_model_weights[0])):
        # Ensure weights are NumPy arrays for each layer
        layer_weight_sum = np.zeros_like(np.array(client_model_weights[0][layer]))
        
        # Perform weighted sum of each client's layer weights
        for i, weights in enumerate(client_model_weights):
            # Convert weights to NumPy arrays before multiplying
            layer_weight_sum += normalized_weights[i] * np.array(weights[layer])
        
        # Add to aggregated weights
        aggregated_weights.append(layer_weight_sum)
    
    # Set the global model weights
    global_model.set_weights(aggregated_weights)
    
    return "Weighted average aggregation completed successfully!"


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

from flask import Flask, request
import base64

@app.route('/receive_model', methods=['POST'])
def receive_model():
    global client_model_weights
    received_weights_encoded = request.json['model_weights']    
    received_weights_bytes = base64.b64decode(received_weights_encoded)
    decrypted_weights_json = fernet.decrypt(received_weights_bytes).decode('utf-8')
    received_weights = json.loads(decrypted_weights_json)
    
    client_model_weights.append(received_weights)
    
    if len(client_model_weights) > 4:
        client_model_weights.pop(0)
    
    return "Model weights received from client."


@app.route('/send_global_weights', methods=['GET'])
def send_global_weights():
    global global_model 
    weights_as_list = [w.tolist() for w in global_model.get_weights()]
    weights_json = json.dumps(weights_as_list)    
    encrypted_data = fernet.encrypt(weights_json.encode())
    encoded_data = base64.b64encode(encrypted_data).decode('utf-8')  # Convert to string    
    return {"global_weights" : encoded_data}

if __name__ == "__main__":
    app.run(debug=True, port=5000)
