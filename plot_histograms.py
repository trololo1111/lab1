"""Гистограммы строк и столбцов случайной матрицы из задания 4."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from grader_contracts.numpy_tasks import RandomMatrixInput
from numpy_tasks import matrix_statistics


def plot_histograms(matrix: np.ndarray, bins: int = 10) -> tuple[plt.Figure, plt.Figure]:
    """Вернуть две фигуры: гистограммы каждой строки и каждого столбца."""
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or 0 in matrix.shape:
        raise ValueError("Ожидается непустая двумерная матрица")
    figures = []
    for vectors, label in ((matrix, "Row"), (matrix.T, "Column")):
        figure, axes = plt.subplots(len(vectors), 1, figsize=(8, 2.5 * len(vectors)), squeeze=False)
        for index, (values, axis) in enumerate(zip(vectors, axes[:, 0]), start=1):
            plt.sca(axis)
            plt.hist(values, bins=bins, color="#377eb8", edgecolor="white")
            axis.set_title(f"{label} {index}")
            axis.set_xlabel("Value")
            axis.set_ylabel("Count")
        figure.tight_layout()
        figures.append(figure)
    return figures[0], figures[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--mean", type=float, default=0.0)
    parser.add_argument("--std", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, help="Сохранить PNG вместо открытия окон")
    args = parser.parse_args()
    statistics = matrix_statistics(RandomMatrixInput(args.rows, args.columns, args.mean, args.std, args.seed))
    row_figure, column_figure = plot_histograms(statistics.matrix, args.bins)
    if args.output_dir is None:
        plt.show()
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        row_figure.savefig(args.output_dir / "rows.png", dpi=150)
        column_figure.savefig(args.output_dir / "columns.png", dpi=150)
        plt.close(row_figure)
        plt.close(column_figure)


if __name__ == "__main__":
    main()
