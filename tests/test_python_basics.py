import unittest

import numpy as np

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput
from python_basics import (
    count_one_bits, count_vowels, has_unique_characters, is_balanced_number,
    mse, multiplicative_persistence, prime_factorization, pyramid,
)


class PythonBasicsTests(unittest.TestCase):
    def test_count_vowels(self):
        for text, expected in [("", 0), ("AEiou", 5), ("Hello World", 3), ("rhythm", 0), ("aAa", 3)]:
            with self.subTest(text=text):
                self.assertEqual(count_vowels(TextInput(text)), expected)

    def test_unique_characters(self):
        for text, expected in [("", True), ("a", True), ("abc", True), ("aba", False), ("aA", True), (" ", True), ("  ", False)]:
            with self.subTest(text=text):
                self.assertIs(has_unique_characters(TextInput(text)), expected)

    def test_one_bits(self):
        for value, expected in [(0, 0), (1, 1), (7, 3), (42, 3), (2**100, 1), (2**100 - 1, 100)]:
            with self.subTest(value=value):
                self.assertEqual(count_one_bits(PositiveIntegerInput(value)), expected)

    def test_persistence(self):
        for value, expected in [(39, 3), (4, 0), (999, 4), (10, 1), (77, 4), (0, 0)]:
            with self.subTest(value=value):
                self.assertEqual(multiplicative_persistence(PositiveIntegerInput(value)), expected)

    def test_mse(self):
        for predicted, expected, error in [([1, 2, 3], [1, 2, 3], 0.0), ([1, 2, 3], [2, 4, 6], 14 / 3), ([-1], [2], 9.0), ([0, 0], [1, 0], 0.5)]:
            with self.subTest(predicted=predicted, expected=expected):
                result = mse(VectorPairInput(predicted, expected))
                self.assertIsInstance(result, float)
                self.assertAlmostEqual(result, error)

    def test_mse_large_integers(self):
        self.assertEqual(mse(VectorPairInput(np.array([2**62]), np.array([-2**62]))), float(2**126))

    def test_mse_invalid_vectors(self):
        for predicted, expected in [([], []), ([1], []), ([1, 2], [1])]:
            with self.subTest(predicted=predicted, expected=expected), self.assertRaises(ValueError):
                mse(VectorPairInput(predicted, expected))

    def test_prime_factorization(self):
        for value, expected in [(86240, "(2**5)(5)(7**2)(11)"), (1, ""), (2, "(2)"), (97, "(97)"), (64, "(2**6)"), (49, "(7**2)"), (30, "(2)(3)(5)")]:
            with self.subTest(value=value):
                self.assertEqual(prime_factorization(PositiveIntegerInput(value)), expected)

    def test_pyramid(self):
        for value, expected in [(1, 1), (5, 2), (14, 3), (30, 4), (2, "It is impossible"), (15, "It is impossible")]:
            with self.subTest(value=value):
                self.assertEqual(pyramid(PositiveIntegerInput(value)), expected)

    def test_pyramid_large_number(self):
        height = 10**6
        cubes = height * (height + 1) * (2 * height + 1) // 6
        self.assertEqual(pyramid(PositiveIntegerInput(cubes)), height)
        self.assertEqual(pyramid(PositiveIntegerInput(cubes + 1)), "It is impossible")

    def test_balanced_number(self):
        for value, expected in [(1234006, True), (123456, False), (7, True), (12, True), (101, True), (123, False), (1231, True), (1232, False), (0, True)]:
            with self.subTest(value=value):
                self.assertIs(is_balanced_number(PositiveIntegerInput(value)), expected)

    def test_invalid_negative_numbers(self):
        for function in [count_one_bits, multiplicative_persistence, prime_factorization, pyramid, is_balanced_number]:
            with self.subTest(function=function.__name__), self.assertRaises(ValueError):
                function(PositiveIntegerInput(-1))


if __name__ == "__main__":
    unittest.main()
