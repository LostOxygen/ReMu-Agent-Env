import unittest

from ai_wars.networking.client_entry import ClientEntry

class TestClientEntry(unittest.TestCase):

	def test_as_tuple(self):
		expected = ("127.0.0.1", 1234)

		c = ClientEntry("127.0.0.1", 1234)
		actual = c.as_tuple()

		self.assertEqual(expected, actual)

	def test___eq__(self):
		c1 = ClientEntry("127.0.0.1", 1234)
		c2 = ClientEntry("127.0.0.1", 1234)

		self.assertTrue(c1 == c2)

	def test___str__(self):
		expected = "127.0.0.1:1234"

		c = ClientEntry("127.0.0.1", 1234)
		actual = str(c)

		self.assertEqual(expected, actual)

