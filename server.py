"""main hook to start the game"""
import argparse
import logging
from ai_wars.server.game_class import GameClass
from ai_wars.server.server_modus import Realtime, TrainingMode

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("--port", "-p", help="specify port on which the server runs",
						type=int, default=1337)
	parser.add_argument("--addr", "-a", help="specify the network address on which the server runs",
						type=str, default="127.0.0.1")
	parser.add_argument("--training_mode", help="sets the server to training mode which updates "
						"once all clients provide their action", action="store_true")
	parser.add_argument("--verbose", "-v", help="enable logging mode for the server",
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

	mode = TrainingMode() if args.training_mode else Realtime()
	server = GameClass(mode, addr=args.addr, port=args.port)
	server.loop()
