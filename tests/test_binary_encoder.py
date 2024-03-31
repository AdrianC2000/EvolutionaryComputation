import unittest

import numpy as np

from binary_encoder import BinaryEncoder


class TestBinaryEncoder(unittest.TestCase):

    def test_should_encode_whole_population(self):
        # given
        encoder = BinaryEncoder(precision=5, limit_start=10, limit_end=20)
        population_size = 5
        variables_number = 10
        population = np.random.uniform(10, 20, size=(population_size, variables_number))

        # when
        encoded_population = encoder.encode_population(population)

        # then
        self.assertEqual(encoded_population.shape, (population_size, variables_number))
        assert np.all(np.vectorize(self._is_binary_string)(encoded_population))

    @staticmethod
    def _is_binary_string(s):
        return set(s) == {'0', '1'}

    def test_should_encode_and_decode_floats_when_positive_numbers(self):
        encoder = BinaryEncoder(precision=5, limit_start=10, limit_end=20)
        numbers_to_test = [10.0, 12.25, 14.5, 16.75, 18.23]

        for number in numbers_to_test:
            with self.subTest(number=number):
                encoded_binary = encoder.encode_to_binary(number)
                decoded_number = encoder.decode_from_binary(encoded_binary)
                self.assertAlmostEqual(decoded_number, number, places=4)

    def test_should_encode_and_decode_exact_floats_when_high_precision(self):
        encoder = BinaryEncoder(precision=15, limit_start=10, limit_end=20)
        numbers_to_test = [10.0, 12.25, 14.5, 16.75, 18.23]

        for number in numbers_to_test:
            with self.subTest(number=number):
                encoded_binary = encoder.encode_to_binary(number)
                decoded_number = encoder.decode_from_binary(encoded_binary)
                self.assertEqual(decoded_number, number)

    def test_should_encode_and_decode_floats_when_negative_numbers(self):
        encoder = BinaryEncoder(precision=5, limit_start=-20, limit_end=-10)
        numbers_to_test = [-18.23, -16.75, -14.5, -12.25, -10]

        for number in numbers_to_test:
            with self.subTest(number=number):
                encoded_binary = encoder.encode_to_binary(number)
                decoded_number = encoder.decode_from_binary(encoded_binary)
                self.assertAlmostEqual(decoded_number, number, places=4)
