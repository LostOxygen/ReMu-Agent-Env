class AgentNotFoundException(Exception):

	def __init__(self, name):
		super().__init__(f"Model with name {name} not found!")
