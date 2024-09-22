import pandas as pd
from sklearn.model_selection import train_test_split


data = pd.read_csv("/Users/anand/Documents/College/Sem7/CryptoProject/Helpers/detect_dataset.csv")
X, y = data[['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']], data['Output (S)']


X_temp1, X_temp2, y_temp1, y_temp2 = train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)
X_client1, X_client2, y_client1, y_client2 = train_test_split(X_temp1, y_temp1, test_size=0.5, stratify=y_temp1, random_state=42)
X_client3, X_client4, y_client3, y_client4 = train_test_split(X_temp2, y_temp2, test_size=0.5, stratify=y_temp2, random_state=42)


client1_data = pd.concat([X_client1, y_client1], axis=1)
client2_data = pd.concat([X_client2, y_client2], axis=1)
client3_data = pd.concat([X_client3, y_client3], axis=1)
client4_data = pd.concat([X_client4, y_client4], axis=1)

client1_data.to_csv('client1_data.csv', index=False)
client2_data.to_csv('client2_data.csv', index=False)
client3_data.to_csv('client3_data.csv', index=False)
client4_data.to_csv('client4_data.csv', index=False)

print("Data split and saved for 4 clients.")
