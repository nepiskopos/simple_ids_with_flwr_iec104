import argparse
import ipaddress
import numpy as np
import os
import pandas as pd

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler

import flwr as fl


if __name__ == "__main__" :
	parser = argparse.ArgumentParser(description='Flower straggler / client implementation')
	parser.add_argument("-a", "--address", help="Aggregator server's IP address", default="127.0.0.1")
	parser.add_argument("-p", "--port", help="Aggregator server's serving port", default=8000, type=int)
	parser.add_argument("-i", "--id", help="client ID", default=0, type=int)
	args = parser.parse_args()

	try:
		ipaddress.ip_address(args.address)
	except ValueError:
		sys.exit(f"Wrong IP address: {args.address}")
	if args.port < 0 or args.port > 65535:
		sys.exit(f"Wrong serving port: {args.port}")
	if args.id < 1:
		sys.exit(f"Wrong client ID: {args.id}")

	# Make TensorFlow log less verbose
	os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


	# Load train data
	df = pd.read_csv(f'/root/client_data_{args.id}.csv')
	X_train = df.drop(columns=['y'])
	y_train = df['y']

	# Load test data
	df = pd.read_csv(f'/root/test_data.csv')
	X_test = df.drop(columns=['y'])
	y_test = df['y']

	# Replace NaN and INF values with zeros in feature data
	if np.any(np.isnan(X_train)):
		X_train.replace([np.nan, -np.nan], 0, inplace=True)
	if np.any(np.isnan(X_test)):
		X_test.replace([np.nan, -np.nan], 0, inplace=True)
	if not np.all(np.isfinite(X_train)):
		X_train.replace([np.inf, -np.inf], 0, inplace=True)
	if not np.all(np.isfinite(X_test)):
		X_test.replace([np.inf, -np.inf], 0, inplace=True)

	# Scale feature values for input data normalization
	scaler = MinMaxScaler()

	X_train = scaler.fit_transform(X_train.to_numpy())
	y_train = y_train.to_numpy()
	X_test = scaler.transform(X_test.to_numpy())
	y_test = y_test.to_numpy()

	# Use one-hot-vectors for label representation
	y_cat_train = to_categorical(y_train)
	y_cat_test = to_categorical(y_test)


	# Define a MLP model
	model = tf.keras.models.Sequential([
		tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)),
		tf.keras.layers.Dense(units=50, activation='relu'),
		tf.keras.layers.Dropout(0.2),
		tf.keras.layers.Dense(units=y_cat_test.shape[1], activation='softmax')
	])

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)


	# Define Flower client
	class Client(fl.client.NumPyClient):
		def get_parameters(self, config):
			return model.get_weights()

		def fit(self, parameters, config):
			model.set_weights(parameters)
			model.fit(X_train, y_cat_train, epochs=200, validation_data=(X_test, y_cat_test), batch_size=32, callbacks=[early_stop])
			return model.get_weights(), len(X_train), {}

		def evaluate(self, parameters, config):
			model.set_weights(parameters)
			loss, accuracy = model.evaluate(X_test, y_cat_test)
			return loss, len(X_test), {"accuracy": accuracy, "loss": loss}


	# Start Flower straggler and initiate communication with the Flower aggretation server
	fl.client.start_numpy_client(server_address=f"{args.address}:{args.port}", client=Client())


	# Evaluate model performance
	print(f"Client {args.id}: Loss, Accuracy: ", model.evaluate(X_test, y_cat_test))
	print(f"Client {args.id}: F1-score: ", f1_score(y_test, np.argmax(model.predict(X_test), axis=1), average='weighted'))
