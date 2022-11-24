import argparse
import ipaddress
import os
import pandas as pd

import flwr as fl
import tensorflow as tf


if __name__ == "__main__" :
	parser = argparse.ArgumentParser(description='Flower client')
	parser.add_argument("-a", "--address", help="server's IP address", default="0.0.0.0")
	parser.add_argument("-p", "--port", help="server's serving port", default=8000, type=int)
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

	# Load data
	df_train = pd.read_csv(f'/root/client_data_{args.id-1}.csv')
	df_test = pd.read_csv(f'/root/test_data.csv')

	# Split data into X and Y
	x_train = df_train.drop(columns=['y']).to_numpy()
	y_train = df_train['y'].to_numpy()
	x_test = df_test.drop(columns=['y']).to_numpy()
	y_test = df_test['y'].to_numpy()

	# Define model and
	model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(x_train.shape[1],)),
        tf.keras.layers.Dense(y_train.max()+1, kernel_initializer='zeros', activation='relu'),
        tf.keras.layers.Softmax(),
    ])

	model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

	# Define Flower client
	class Client(fl.client.NumPyClient):
		def get_parameters(self, config):
			return model.get_weights()

		def fit(self, parameters, config):
			model.set_weights(parameters)
			model.fit(x_train, y_train, epochs=1, batch_size=32)
			return model.get_weights(), len(x_train), {}

		def evaluate(self, parameters, config):
			model.set_weights(parameters)
			loss, accuracy = model.evaluate(x_test, y_test)
			return loss, len(x_test), {"accuracy": accuracy}


	# Start Flower client
	fl.client.start_numpy_client(server_address=f"{args.address}:{args.port}", client=Client())
