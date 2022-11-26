import argparse
import ipaddress

import flwr as fl


if __name__ == "__main__" :
	parser = argparse.ArgumentParser(description='Flower aggregator server implementation')
	parser.add_argument("-a", "--address", help="IP address", default="0.0.0.0")
	parser.add_argument("-p", "--port", help="Serving port", default=8000, type=int)
	parser.add_argument("-r", "--rounds", help="Number of training and aggregation rounds", default=3, type=int)
	args = parser.parse_args()

	try:
		ipaddress.ip_address(args.address)
	except ValueError:
		sys.exit(f"Wrong IP address: {args.address}")
	if args.port < 0 or args.port > 65535:
		sys.exit(f"Wrong serving port: {args.port}")
	if args.rounds < 0:
		sys.exit(f"Wrong number of rounds: {args.rounds}")


	# Start Flower aggregation and distribution server
	fl.server.start_server(
		server_address=f"{args.address}:{args.port}",
		config=fl.server.ServerConfig(num_rounds=args.rounds),
	)
