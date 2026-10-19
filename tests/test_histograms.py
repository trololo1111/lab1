import unittest

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from plot_histograms import plot_histograms


class HistogramTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_histograms_for_each_row_and_column(self):
        rows, columns = plot_histograms(np.array([[1, 2, 2], [3, 3, 4]]), bins=2)
        self.assertEqual(len(rows.axes), 2)
        self.assertEqual(len(columns.axes), 3)
        for axis in rows.axes:
            self.assertEqual(sum(bar.get_height() for bar in axis.patches), 3)
        for axis in columns.axes:
            self.assertEqual(sum(bar.get_height() for bar in axis.patches), 2)

    def test_single_cell(self):
        rows, columns = plot_histograms(np.array([[5]]), bins=1)
        self.assertEqual(rows.axes[0].patches[0].get_height(), 1)
        self.assertEqual(columns.axes[0].patches[0].get_height(), 1)

    def test_empty_matrix(self):
        with self.assertRaises(ValueError):
            plot_histograms(np.empty((0, 3)))


if __name__ == "__main__":
    unittest.main()
