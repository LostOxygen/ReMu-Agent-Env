from typing import Tuple

class ClientEntry:
	'''
	Data class for a client
	'''

	addr: str
	port: int

	def __init__(self, addr: str, port: int):
		self.addr = addr
		self.port = port

	def as_tuple(self) -> Tuple[str, int]:
		'''
		Returns tuple representation of a client.

		Returns:
			(address of the client, port of the client)
		'''

		return (self.addr, self.port)

	def __eq__(self, other) -> bool:
		if not isinstance(other, ClientEntry):
			return False

		return self.addr == other.addr \
			and self.port == other.port

	def __hash__(self) -> int:
		return hash(self.as_tuple())

	def __str__(self) -> str:
		return f"{self.addr}:{self.port}"
