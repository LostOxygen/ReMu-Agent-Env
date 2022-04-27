import gzip

from .layer import Layer

class GzipCompression(Layer):
	'''
	Applies gzip compression and decrompression to the data
	'''

	def forward(self, data: bytes) -> bytes:
		return gzip.compress(data)

	def backward(self, data: bytes) -> bytes:
		return gzip.decompress(data)
