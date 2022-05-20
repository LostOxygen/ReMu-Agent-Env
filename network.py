"""main hook to start the game"""
import multiprocessing
import argparse
import logging
from ai_wars.client.player import Player
from ai_wars.client.dqn_behavior import DqnBehavior


def spawn_network(model_name: str, addr: str, port: int) -> None:
	"""spawn a network player"""
	player = Player(model_name, addr, port, DqnBehavior(model_name, None))
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
	if args.n_models > 1:
		for i in range(args.n_models):
			name = f"{args.name[0]}_{i}"
			logging.info("Spawning model with name: %s", name)
			model_thread = multiprocessing.Process(target=spawn_network,
                                          args=(name, args.addr, args.port))
			model_thread.start()
	else:
		for name in args.name:
			logging.info("Spawning model with name: %s", name)
			model_thread = multiprocessing.Process(target=spawn_network,
                                          args=(name, args.addr, args.port))
			model_thread.start()
