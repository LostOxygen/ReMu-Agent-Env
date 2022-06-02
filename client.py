"""main hook to start the game"""
import sys
import argparse
import logging
from ai_wars.client.player import Player
from ai_wars.client.game_gui import GameGUI


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--name", "-n", help="specify the player name to connect with",
						type=str)
	parser.add_argument("--spectate", help="enter the spectate mode", action="store_true")
	parser.add_argument("--port", "-p", help="specify port on which the client connects",
						type=int, default=1337)
	parser.add_argument("--addr", "-a", help="specify the network addr. on which the client connects",
						type=str, default="127.0.0.1")
	parser.add_argument("--verbose", "-v", help="enable logging mode for the client",
						action="store_true", default=False)
	args = parser.parse_args()

	if args.name is None and not args.spectate:
		print("Error: Either --name <name> or --spectate must be specified!")
		sys.exit(0)

	if args.name is not None and args.spectate:
		print("Error: Arguments --name <name> and --spectate may not specified at the same time!")
		sys.exit(0)

	player_name = "spectator" if args.spectate else args.name

	if args.verbose:
		logging.basicConfig(level=logging.DEBUG,
						format="%(asctime)-8s %(levelname)-8s %(message)s",
						datefmt="%H:%M:%S")
	else:
		logging.basicConfig(level=logging.CRITICAL,
						format="%(asctime)-8s %(levelname)-8s %(message)s",
						datefmt="%H:%M:%S")

	player = Player(player_name, args.addr, args.port, GameGUI())
	player.loop()
