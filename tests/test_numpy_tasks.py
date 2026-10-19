import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)
from numpy_tasks import (
    analyze_time_series, binarize, chess, draw_ellipse, draw_rectangle,
    matrix_statistics, one_hot, sum_prod, unique_columns, unique_rows,
)


class NumpyTasksTests(unittest.TestCase):
    def test_sum_prod(self):
        matrices = np.array([[[1, 2], [3, 4]], [[2, 0], [0, 2]]])
        vectors = np.array([[[1], [2]], [[3], [4]]])
        original_matrices, original_vectors = matrices.copy(), vectors.copy()
        assert_array_equal(sum_prod(MatrixVectorBatchInput(matrices, vectors)), [[11], [19]])
        assert_array_equal(matrices, original_matrices)
        assert_array_equal(vectors, original_vectors)

    def test_sum_prod_single_and_empty_batches(self):
        assert_allclose(sum_prod(MatrixVectorBatchInput([[[2.5]]], [[[4]]])), [[10]])
        result = sum_prod(MatrixVectorBatchInput(np.empty((0, 2, 2)), np.empty((0, 2, 1))))
        assert_array_equal(result, [[0], [0]])

    def test_sum_prod_invalid_shapes(self):
        for matrices, vectors in [(np.zeros((2, 3, 2)), np.zeros((2, 3, 1))), (np.zeros((2, 2, 2)), np.zeros((1, 2, 1)))]:
            with self.subTest(shape=matrices.shape), self.assertRaises(ValueError):
                sum_prod(MatrixVectorBatchInput(matrices, vectors))

    def test_binarize_threshold_and_copy(self):
        matrix = np.array([[-1, 0.5, 0.6], [2, 0, 0.5]])
        original = matrix.copy()
        result = binarize(BinarizeInput(matrix))
        assert_array_equal(result, [[0, 0, 1], [1, 0, 0]])
        assert_array_equal(matrix, original)
        self.assertFalse(np.shares_memory(matrix, result))

    def test_binarize_empty_and_custom_threshold(self):
        self.assertEqual(binarize(BinarizeInput(np.empty((0, 3)))).shape, (0, 3))
        assert_array_equal(binarize(BinarizeInput([[-2, -1, 0]], -1)), [[0, 0, 1]])

    def test_unique_rows(self):
        self.assertEqual(unique_rows(MatrixInput([[3, 1, 3], [2, 2, 2]])), [[1, 3], [2]])
        self.assertEqual(unique_rows(MatrixInput([[1.5, -2, 1.5]])), [[-2.0, 1.5]])
        self.assertEqual(unique_rows(MatrixInput(np.empty((0, 3)))), [])
        self.assertEqual(unique_rows(MatrixInput(np.empty((2, 0)))), [[], []])

    def test_unique_columns(self):
        self.assertEqual(unique_columns(MatrixInput([[3, 1, 3], [2, 1, 2], [3, 2, 3]])), [[2, 3], [1, 2], [2, 3]])
        self.assertEqual(unique_columns(MatrixInput([[1.5], [-2], [1.5]])), [[-2.0, 1.5]])
        self.assertEqual(unique_columns(MatrixInput(np.empty((0, 2)))), [[], []])
        self.assertEqual(unique_columns(MatrixInput(np.empty((2, 0)))), [])

    def test_matrix_statistics(self):
        result = matrix_statistics(RandomMatrixInput(3, 4, mean=2.0, std=1.5, seed=7))
        self.assertIsInstance(result, MatrixStatistics)
        self.assertEqual(result.matrix.shape, (3, 4))
        assert_allclose(result.row_means, [sum(row) / 4 for row in result.matrix])
        assert_allclose(result.column_means, [sum(column) / 3 for column in result.matrix.T])
        assert_allclose(result.row_variances, [sum((row - mean) ** 2) / 4 for row, mean in zip(result.matrix, result.row_means)])
        assert_allclose(result.column_variances, [sum((column - mean) ** 2) / 3 for column, mean in zip(result.matrix.T, result.column_means)])

    def test_matrix_statistics_seed(self):
        first = matrix_statistics(RandomMatrixInput(2, 3, seed=42)).matrix
        second = matrix_statistics(RandomMatrixInput(2, 3, seed=42)).matrix
        third = matrix_statistics(RandomMatrixInput(2, 3, seed=43)).matrix
        assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, third))

    def test_matrix_statistics_constant_and_single_cell(self):
        for rows, columns in [(2, 3), (1, 1)]:
            with self.subTest(shape=(rows, columns)):
                result = matrix_statistics(RandomMatrixInput(rows, columns, mean=5, std=0))
                assert_array_equal(result.matrix, np.full((rows, columns), 5))
                assert_array_equal(result.row_means, np.full(rows, 5))
                assert_array_equal(result.column_means, np.full(columns, 5))
                assert_array_equal(result.row_variances, np.zeros(rows))
                assert_array_equal(result.column_variances, np.zeros(columns))

    def test_matrix_statistics_invalid_parameters(self):
        for data in [RandomMatrixInput(0, 3), RandomMatrixInput(3, 0), RandomMatrixInput(2, 2, std=-1)]:
            with self.subTest(data=data), self.assertRaises(ValueError):
                matrix_statistics(data)

    def test_chess_example(self):
        assert_array_equal(chess(ChessInput(3, 4, 0, 1)), [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]])

    def test_chess_edges(self):
        assert_array_equal(chess(ChessInput(1, 1, 2.5, -1)), [[2.5]])
        assert_array_equal(chess(ChessInput(3, 1, 2.5, -1)), [[2.5], [-1], [2.5]])
        assert_array_equal(chess(ChessInput(1, 3, 7, 7)), [[7, 7, 7]])
        self.assertEqual(chess(ChessInput(0, 4, 0, 1)).shape, (0, 4))

    def test_rectangle_position_and_colors(self):
        result = draw_rectangle(RectangleInput(3, 1, 5, 7, (255, 0, 0), (10, 20, 30)))
        expected = np.full((5, 7, 3), (10, 20, 30), dtype=np.uint8)
        expected[2, 2:5] = (255, 0, 0)
        assert_array_equal(result, expected)
        self.assertEqual(result.dtype, np.uint8)

    def test_rectangle_zero_and_oversized(self):
        blank = draw_rectangle(RectangleInput(0, 2, 4, 4, (255, 0, 0), (0, 0, 0)))
        assert_array_equal(blank, np.zeros((4, 4, 3), dtype=np.uint8))
        filled = draw_rectangle(RectangleInput(10, 10, 3, 4, (1, 2, 3), (0, 0, 0)))
        assert_array_equal(filled, np.full((3, 4, 3), (1, 2, 3), dtype=np.uint8))

    def test_rectangle_even_size(self):
        result = draw_rectangle(RectangleInput(2, 2, 4, 4, (255, 255, 255), (0, 0, 0)))
        assert_array_equal(result[:, :, 0], [[0, 0, 0, 0], [0, 255, 255, 0], [0, 255, 255, 0], [0, 0, 0, 0]])

    def test_ellipse_circle_and_boundary(self):
        result = draw_ellipse(EllipseInput(1, 1, 5, 5, (1, 2, 3), (10, 20, 30)))
        expected = np.full((5, 5, 3), (10, 20, 30), dtype=np.uint8)
        for row, column in [(1, 2), (2, 1), (2, 2), (2, 3), (3, 2)]:
            expected[row, column] = (1, 2, 3)
        assert_array_equal(result, expected)
        self.assertEqual(result.dtype, np.uint8)

    def test_ellipse_axes(self):
        result = draw_ellipse(EllipseInput(2, 1, 5, 7, (255, 0, 0), (0, 0, 0)))
        expected = np.zeros((5, 7), dtype=np.uint8)
        expected[2, 1:6] = 255
        expected[1, 3] = expected[3, 3] = 255
        assert_array_equal(result[:, :, 0], expected)

    def test_ellipse_even_canvas_symmetry(self):
        result = draw_ellipse(EllipseInput(2, 1, 4, 6, (255, 0, 0), (0, 0, 0)))
        assert_array_equal(result, result[::-1])
        assert_array_equal(result, result[:, ::-1])

    def test_ellipse_invalid_axes(self):
        with self.assertRaises(ValueError):
            draw_ellipse(EllipseInput(0, 1, 4, 4, (255, 0, 0), (0, 0, 0)))

    def test_ellipse_keeps_exact_boundary_pixels(self):
        result = draw_ellipse(EllipseInput(13, 13, 27, 27, (255, 0, 0), (0, 0, 0)))
        rows, columns = np.indices((27, 27))
        # Точки со смещениями (5, 12) лежат ровно на окружности: 25 + 144 = 169.
        expected = ((rows - 13) ** 2 + (columns - 13) ** 2 <= 169).astype(np.uint8) * 255
        assert_array_equal(result[:, :, 0], expected)

    def test_time_series_statistics(self):
        values = np.array([1, 3, 2, 4, 0])
        result = analyze_time_series(TimeSeriesInput(values, 2))
        self.assertIsInstance(result, TimeSeriesStatistics)
        self.assertEqual(result.mean, 2.0)
        self.assertEqual(result.variance, 2.0)
        self.assertAlmostEqual(result.std, 2**0.5)
        assert_array_equal(result.local_maxima_indices, [1, 3])
        assert_array_equal(result.local_minima_indices, [2])
        assert_allclose(result.moving_average, [2, 2.5, 3, 2])
        assert_array_equal(values, [1, 3, 2, 4, 0])

    def test_time_series_plateau_and_monotone(self):
        for values in [[1, 2, 2, 1], [3, 2, 1], [1, 2, 3], [1, 1, 1]]:
            with self.subTest(values=values):
                result = analyze_time_series(TimeSeriesInput(values, 1))
                self.assertEqual(result.local_maxima_indices.size, 0)
                self.assertEqual(result.local_minima_indices.size, 0)
                assert_allclose(result.moving_average, values)

    def test_time_series_singleton_and_full_window(self):
        singleton = analyze_time_series(TimeSeriesInput([7], 1))
        self.assertEqual(singleton.mean, 7.0)
        self.assertEqual(singleton.variance, 0.0)
        self.assertEqual(singleton.std, 0.0)
        assert_array_equal(singleton.local_maxima_indices, [])
        assert_array_equal(singleton.local_minima_indices, [])
        assert_array_equal(singleton.moving_average, [7])
        full = analyze_time_series(TimeSeriesInput([1, 3, 2], 3))
        assert_allclose(full.moving_average, [2])

    def test_time_series_minimum(self):
        result = analyze_time_series(TimeSeriesInput([3, 1, 3], 2))
        assert_array_equal(result.local_minima_indices, [1])
        assert_array_equal(result.local_maxima_indices, [])

    def test_time_series_invalid_input(self):
        for values, window in [([], 1), ([1, 2], 0), ([1, 2], 3), ([[1, 2]], 1)]:
            with self.subTest(values=values, window=window), self.assertRaises(ValueError):
                analyze_time_series(TimeSeriesInput(values, window))

    def test_time_series_large_integer_extrema(self):
        cases = [
            ([2**53, 2**53 + 1, 2**53], [1], []),
            ([2**53 + 1, 2**53, 2**53 + 1], [], [1]),
            ([2**63 - 2, 2**63 - 1, 2**63 - 2], [1], []),
            ([2**63, 2**63 + 1, 2**63], [1], []),
        ]
        for values, maxima, minima in cases:
            with self.subTest(values=values):
                result = analyze_time_series(TimeSeriesInput(np.array(values), 1))
                assert_array_equal(result.local_maxima_indices, maxima)
                assert_array_equal(result.local_minima_indices, minima)

    def test_time_series_large_integer_variance(self):
        for base in [2**53, 2**63 - 2, 2**63]:
            with self.subTest(base=base):
                result = analyze_time_series(TimeSeriesInput([base, base + 1, base], 2))
                self.assertAlmostEqual(result.variance, 2 / 9)
                self.assertAlmostEqual(result.std, (2 / 9) ** 0.5)

    def test_one_hot_example(self):
        labels = np.array([0, 2, 3, 0])
        result = one_hot(OneHotInput(labels))
        assert_array_equal(result, [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]])
        assert_array_equal(labels, [0, 2, 3, 0])

    def test_one_hot_explicit_class_count(self):
        assert_array_equal(one_hot(OneHotInput([1, 0], 4)), [[0, 1, 0, 0], [1, 0, 0, 0]])
        assert_array_equal(one_hot(OneHotInput([0, 0])), [[1], [1]])
        assert_array_equal(one_hot(OneHotInput(np.array([1], dtype=np.uint8))), [[0, 1]])

    def test_one_hot_empty(self):
        self.assertEqual(one_hot(OneHotInput([])).shape, (0, 0))
        self.assertEqual(one_hot(OneHotInput([], 3)).shape, (0, 3))

    def test_one_hot_invalid_labels(self):
        for data in [OneHotInput([-1]), OneHotInput([1.5]), OneHotInput([2], 2), OneHotInput([], -1), OneHotInput([[0]])]:
            with self.subTest(data=data), self.assertRaises(ValueError):
                one_hot(data)


if __name__ == "__main__":
    unittest.main()
