"""main hook to start the game"""
import argparse
import logging
import torch

from ai_wars.client.player import Player
from ai_wars.dqn.dqn_behavior import DqnBehavior


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--name", "-n", help="specify the player name to connect with",
						type=str, required=True)
	parser.add_argument("--port", "-p", help="specify port on which the client connects",
						type=int, default=1337)
	parser.add_argument("--addr", "-a", help="specify the network addr. on which the client connects",
						type=str, default="127.0.0.1")
	parser.add_argument("--verbose", "-l", help="enable logging mode for the client",
						action="store_true", default=False)
	parser.add_argument("--model_type", "-m", help="Specify the model type ('linear' or 'lstm')",
						type=str, required=True)
	args = parser.parse_args()

	if args.verbose:
		logging.basicConfig(level=logging.DEBUG,
						format="%(asctime)-8s %(levelname)-8s %(message)s",
						datefmt="%H:%M:%S")
	else:
		logging.basicConfig(level=logging.CRITICAL,
						format="%(asctime)-8s %(levelname)-8s %(message)s",
						datefmt="%H:%M:%S")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	behavior = DqnBehavior(args.name, args.model_type, device)
	p = Player(args.name, args.addr, args.port, behavior)
	p.loop()
