from cryptography.fernet import Fernet
import json

# Generate a key and create a Fernet instance
key = Fernet.generate_key()
print(key.decode())
fernet = Fernet(key)

# Create a dictionary
data = {"weights": "123"}

# Convert the dictionary to a JSON string
data_json = json.dumps(data)

# Encrypt the JSON string
encrypted_data = fernet.encrypt(data_json.encode())

print("Encrypted:", encrypted_data)

# If you want to decrypt it later
decrypted_data = fernet.decrypt(encrypted_data).decode()
decrypted_dict = json.loads(decrypted_data)

print("Decrypted:", decrypted_dict)

