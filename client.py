"""main hook to start the game"""
import sys
from ai_wars.client.game_class import GameClass


if __name__ == "__main__":
	client = GameClass(sys.argv[1])
	client.main_loop()
