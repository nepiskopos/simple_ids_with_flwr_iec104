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

import flwr as fl


def fit_round(server_round: int) -> Dict:
	"""Send round number to client."""
	return {"server_round": server_round}

def get_evaluate_fn(model: Sequential):
	"""Return an evaluation function for server-side evaluation."""

	def evaluate(
		server_round: int,
		parameters: fl.common.NDArrays,
		config: Dict[str, fl.common.Scalar],
	):
		# Update model with the latest parameters
		model.set_weights(parameters)
		loss, accuracy = model.evaluate(X_test_scaled, y_test_cat)
		f1 = f1_score(y_test, np.argmax(model.predict(X_test_scaled), axis=1), average='weighted')

		return loss, {"accuracy": accuracy, "f1-score": f1}

	return evaluate


if __name__ == "__main__" :
	parser = argparse.ArgumentParser(description='Flower aggregator server implementation')
	parser.add_argument("-a", "--address", help="IP address", default="0.0.0.0")
	parser.add_argument("-p", "--port", help="Serving port", default=8000, type=int)
	parser.add_argument("-r", "--rounds", help="Number of training and aggregation rounds", default=20, type=int)
	parser.add_argument("-d", "--dataset", help="dataset directory", default="/root/datasets/federated_datasets/")
	args = parser.parse_args()

	try:
		ipaddress.ip_address(args.address)
	except ValueError:
		sys.exit(f"Wrong IP address: {args.address}")
	if args.port < 0 or args.port > 65535:
		sys.exit(f"Wrong serving port: {args.port}")
	if args.rounds < 0:
		sys.exit(f"Wrong number of rounds: {args.rounds}")
	if not os.path.isdir(args.dataset):
		sys.exit(f"Wrong path to directory with datasets: {args.dataset}")


	# Load train and test data
	df_train = pd.read_csv(os.path.join(args.dataset, 'train_data.csv'))
	df_test = pd.read_csv(os.path.join(args.dataset, 'test_data.csv'))

	# Convert data to arrays
	X_train = df_train.drop(['y'], axis=1).to_numpy()
	X_test = df_test.drop(['y'], axis=1).to_numpy()

	# Convert test data labels to one-hot-vectors
	y_test = df_test['y'].to_numpy()
	y_test_cat = to_categorical(y_test)

	# Scale test data
	scaler = MinMaxScaler()
	scaler.fit(X_train)
	X_test_scaled = scaler.transform(X_test)


	# Define a MLP model
	model = Sequential([
		InputLayer(input_shape=(X_test_scaled.shape[1],)),
		Dense(units=50, activation='relu'),
		Dropout(0.2),
		Dense(units=y_test_cat.shape[1], activation='softmax')
	])

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


	# Define a FL strategy
	strategy = fl.server.strategy.FedAvg(
		min_available_clients=3,
		evaluate_fn=get_evaluate_fn(model),
		on_fit_config_fn=fit_round,
	)

	# Start Flower aggregation and distribution server
	fl.server.start_server(
		server_address=f"{args.address}:{args.port}",
		strategy=strategy,
		config=fl.server.ServerConfig(num_rounds=args.rounds),
	)
