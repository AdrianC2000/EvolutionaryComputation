import unittest
from binary_encoder import BinaryEncoder


class TestBinaryEncoder(unittest.TestCase):

    def test_should_encode_and_decode_floats_when_positive_numbers(self):
        encoder = BinaryEncoder(precision=5, limit_start=567, limit_end=24324)
        numbers_to_test = [10.0, 12.25, 14.5, 16.75, 18.23]

        for number in numbers_to_test:
            with self.subTest(number=number):
                encoded_binary = encoder.encode_to_binary(number)
                decoded_number = encoder.decode_from_binary(encoded_binary)
                self.assertAlmostEqual(decoded_number, number, places=4)

    def test_should_encode_and_decode_floats_when_negative_numbers(self):
        encoder = BinaryEncoder(precision=5, limit_start=-20, limit_end=-10)
        numbers_to_test = [-18.23, -16.75, -14.5, -12.25, -10]

        for number in numbers_to_test:
            with self.subTest(number=number):
                encoded_binary = encoder.encode_to_binary(number)
                decoded_number = encoder.decode_from_binary(encoded_binary)
                self.assertAlmostEqual(decoded_number, number, places=4)
