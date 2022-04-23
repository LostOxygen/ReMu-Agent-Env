"""main hook to start the game"""
from ai_wars.game_class import GameClass

if __name__ == "__main__":
	ai_wars = GameClass()
	ai_wars.main_loop()
