"""main hook to start the game"""
import argparse
import multiprocessing
import socket
import datetime
import os
import logging
import torch

from ai_wars.client.player import Player
from ai_wars.dqn.dqn_behavior import DqnBehavior


def spawn_network(model_name: str, addr: str, port: int, model_type: str, device: str) -> None:
	"""spawn a network player"""
	player = Player(model_name, addr, port, DqnBehavior(model_name, model_type, device))
	player.loop()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--name", "-n", help="specify the player name to connect with",
						nargs="+", type=str, required=True)
	parser.add_argument("--port", "-p", help="specify port on which the client connects",
						type=int, default=1337)
	parser.add_argument("--addr", "-a", help="specify the network addr. on which the client connects",
						type=str, default="127.0.0.1")
	parser.add_argument("--verbose", "-l", help="enable logging mode for the client",
						action="store_true", default=False)
	parser.add_argument("--model_type", "-m", help="Specify the model type ('linear' or 'lstm')",
						type=str, required=True)
	parser.add_argument("--device", "-d", help="Specify the device for the computations",
                     	type=str, default="cuda:0")
	parser.add_argument("--n_models", "-rm", help="spawns N models simultaneously",
                     	type=int, default=1)

	args = parser.parse_args()

	if args.verbose:
		logging.basicConfig(level=logging.DEBUG,
						format="%(asctime)-8s %(levelname)-8s %(message)s",
						datefmt="%H:%M:%S")
	else:
		logging.basicConfig(level=logging.CRITICAL,
						format="%(asctime)-8s %(levelname)-8s %(message)s",
						datefmt="%H:%M:%S")

	device = args.device
	if not torch.cuda.is_available():
		# overwrite the device if no GPU is available
		device = "cpu"

	logging.info("Time: %s", datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
	logging.info(
		"System: %s CPU cores with %s threads and %s GPUs on %s",
       	    torch.get_num_threads(), os.cpu_count(), torch.cuda.device_count(), socket.gethostname()
	)
	logging.info("Using device: %s", device)

	if args.n_models > 1:
		for i in range(args.n_models):
			name = f"{args.name[0]}_{i}"
			logging.info("Spawning model with name: %s", name)
			model_thread = multiprocessing.Process(target=spawn_network,
                                        args=(name, args.addr, args.port, args.model_type, device))
			model_thread.start()
	else:
		for name in args.name:
			logging.info("Spawning model with name: %s", name)
			model_thread = multiprocessing.Process(target=spawn_network,
                                        args=(name, args.addr, args.port, args.model_type, device))
			model_thread.start()
