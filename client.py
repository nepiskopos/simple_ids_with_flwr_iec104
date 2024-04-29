import argparse
import ipaddress
import os
import sys

from typing import Dict

import numpy as np
import pandas as pd

from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

import flwr as fl


if __name__ == "__main__" :
	parser = argparse.ArgumentParser(description='Flower straggler / client implementation')
	parser.add_argument("-a", "--address", help="Aggregator server's IP address", default="127.0.0.1")
	parser.add_argument("-p", "--port", help="Aggregator server's serving port", default=8000, type=int)
	parser.add_argument("-i", "--id", help="client ID", default=0, type=int)
	parser.add_argument("-d", "--dataset", help="dataset directory", default="/root/datasets/federated_datasets/")
	args = parser.parse_args()

	try:
		ipaddress.ip_address(args.address)
	except ValueError:
		sys.exit(f"Wrong IP address: {args.address}")
	if args.port < 0 or args.port > 65535:
		sys.exit(f"Wrong serving port: {args.port}")
	if not os.path.isdir(args.dataset):
		sys.exit(f"Wrong path to directory with datasets: {args.dataset}")

	# Make TensorFlow log less verbose
	os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


	# Load train and test data
	df_train = pd.read_csv(os.path.join(args.dataset, f'client_train_data_{args.id}.csv'))
	df_test = pd.read_csv(os.path.join(args.dataset, 'test_data.csv'))

	# Split data into X and y
	X_train = df_train.drop(columns=['y']).to_numpy()
	y_train = df_train['y'].to_numpy()
	X_test = df_test.drop(columns=['y']).to_numpy()
	y_test = df_test['y'].to_numpy()


	# Scale feature values for input data normalization
	scaler = MinMaxScaler()
	X_train_scaled = scaler.fit_transform(X_train)
	X_test_scaled = scaler.transform(X_test)


	# Use one-hot-vectors for label representation
	y_train_cat = to_categorical(y_train)
	y_test_cat = to_categorical(y_test)


	# Define a MLP model
	model = Sequential([
		InputLayer(input_shape=(X_train_scaled.shape[1],)),
		Dense(units=50, activation='relu'),
		Dropout(0.2),
		Dense(units=y_train_cat.shape[1], activation='softmax')
	])

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)


	# Define Flower client
	class Client(fl.client.NumPyClient):
		def get_parameters(self, config):
			return model.get_weights()

		def fit(self, parameters, config):
			model.set_weights(parameters)
			model.fit(X_train_scaled, y_train_cat, epochs=100, validation_data=(X_test_scaled, y_test_cat), batch_size=64, callbacks=[early_stop], verbose=0)
			print(f"Training finished for round {config['server_round']}")
			return model.get_weights(), len(X_train_scaled), {}

		def evaluate(self, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar],):
			model.set_weights(parameters)
			loss, accuracy = model.evaluate(X_test_scaled, y_test_cat)
			f1 = f1_score(y_test, np.argmax(model.predict(X_test_scaled), axis=1), average='weighted')
			return loss, len(X_test_scaled), {"accuracy": accuracy, "f1-score": f1}


	# Start Flower straggler and initiate communication with the Flower aggretation server
	fl.client.start_numpy_client(
		server_address=f"{args.address}:{args.port}",
		client=Client()
	)
