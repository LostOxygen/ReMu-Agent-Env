"""main hook to start the game"""
from ai_wars.server.game_class import GameClass

if __name__ == "__main__":
	server = GameClass()
	server.main_loop()
