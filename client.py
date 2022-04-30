"""main hook to start the game"""
import sys
from ai_wars.client.player import Player
from ai_wars.client.game_gui import GameGUI


if __name__ == "__main__":
	player_name = sys.argv[1]
	player = Player(player_name, "127.0.0.1", 1337, GameGUI())
	player.loop()
