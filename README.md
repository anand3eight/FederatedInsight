# Federated Learning for Global Fault Diagnostic Model

## Overview

This project implements a federated learning approach to develop a global fault diagnostic model. The model is trained across multiple clients while ensuring data privacy and leveraging local datasets.

## Features

- **Federated Learning**: Clients train their models locally and share only model weights with the server.
- **Global Model Aggregation**: The server aggregates weights from all clients to improve the global model.
- **User Interface**: A simple Flask-based UI allows users to train models, evaluate them, and retrieve global weights.

## Technologies Used

- Python
- Flask
- TensorFlow
- NumPy
- Pandas
- scikit-learn

## Project Structure

```
/root
├── /clients             # Client-side implementation
│   ├── client1.py  
│   ├── client2.py   
│   ├── client3.py   
│   ├── client4.py   
├── /server              # Server-side implementation
│   ├── server_code.py    # Server code to aggregate and evaluate model
├── /data                # Directory for dataset files
└── README.md            # Project documentation
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/anand3eight/FederatedInsight.git
   cd FederatedInsight
   ```

## Usage

### Running the Server

1. Navigate to the server directory and run:

   ```bash
   python server.py
   ```

### Running the Clients

1. Navigate to the client directory and run:

   ```bash
   python client1.py
   ```

2. Interact with the Flask UI at `http://localhost:5000`.

### Interactions

- **Train Model**: Train the local model on the client dataset.
- **Evaluate Model**: Evaluate the aggregated global model on a test dataset.
- **Send Global Weights**: Retrieve and apply the global model weights to the local model.

## Contributing

Feel free to submit issues or pull requests if you would like to contribute to this project.

## License

This project is licensed under the MIT License.

