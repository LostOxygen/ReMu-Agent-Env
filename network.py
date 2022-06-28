"""main hook to start the game"""
import argparse
import socket
import datetime
import os
import logging
import signal
import torch

import ai_wars.constants
from ai_wars.client.player import Player
from ai_wars.dqn.dqn_behavior import DqnBehavior, DqnBehaviorTest
from ai_wars.dqn.heuristic_behaviour import HeuristicBehavior


RUNNING = True

def signal_handler(signum, frame):  # pylint: disable=unused-argument
	global RUNNING

	RUNNING = False
	logging.debug("Stopping server threads")

def is_running() -> bool:
	return RUNNING

if __name__ == "__main__":
	signal.signal(signal.SIGINT, signal_handler)

	parser = argparse.ArgumentParser()
	parser.add_argument("--name", "-n", help="specify the player name to connect with",
						type=str, required=True)
	parser.add_argument("--port", "-p", help="specify port on which the client connects",
						type=int, default=1337)
	parser.add_argument("--addr", "-a", help="specify the network addr. on which the client connects",
						type=str, default="127.0.0.1")
	parser.add_argument("--verbose", "-v", help="enable logging mode for the client",
						action="store_true", default=False)
	parser.add_argument("--model_type", "-m", help="Specify the model type ('linear' or 'lstm')",
						type=str, required=True)
	parser.add_argument("--test", help="Runs the networks in testing mode",
						action="store_true", default=False)
	parser.add_argument("--device", "-d", help="Specify the device for the computations",
						type=str, default="cuda:0")
	parser.add_argument("--param_search", "-ps", help="enable hyperparameter search from dictionary",
                     	action="store_true", default=False)

	args = parser.parse_args()

	if args.verbose:
		logging.basicConfig(level=logging.DEBUG,
						format="%(asctime)-8s %(levelname)-8s %(message)s",
						datefmt="%H:%M:%S")
	else:
		logging.basicConfig(level=logging.CRITICAL,
						format="%(asctime)-8s %(levelname)-8s %(message)s",
						datefmt="%H:%M:%S")
	logging.getLogger("matplotlib.font_manager").disabled = True

	if args.param_search:
		ai_wars.constants.PARAM_SEARCH = True

	device = args.device
	if not torch.cuda.is_available():
		# overwrite the device if no GPU is available
		device = "cpu"

	if args.test:
		behavior = DqnBehaviorTest(args.name, args.model_type, device)
	else:
		if args.model_type == "heuristic":
			behavior = HeuristicBehavior(args.name, args.model_type, device)
		else:
			behavior = DqnBehavior(args.name, args.model_type, device)

	logging.info("Time: %s", datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
	logging.info(
		"System: %s CPU cores with %s threads and %s GPUs on %s",
       	    torch.get_num_threads(), os.cpu_count(), torch.cuda.device_count(), socket.gethostname()
	)
	logging.info("Using device: %s", device)
	logging.info("Model type: %s", args.model_type)
	logging.info("Parameter search: %s", args.param_search)
	logging.info("Testing mode: %s", args.test)
	logging.info("Spawning model with name: %s", args.name)

	player = Player(args.name, args.addr, args.port, behavior)
	player.loop(is_running)
