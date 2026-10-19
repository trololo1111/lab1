"""Решения задач на NumPy."""

import numpy as np

from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = np.asarray(data.matrices), np.asarray(data.vectors)
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError("Матрицы должны иметь форму (p, n, n)")
    if vectors.shape != (matrices.shape[0], matrices.shape[1], 1):
        raise ValueError("Векторы должны иметь форму (p, n, 1)")
    return (matrices @ vectors).sum(axis=0)


def binarize(data: BinarizeInput) -> np.ndarray:
    return (np.asarray(data.matrix) > data.threshold).astype(int)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)
    if matrix.ndim != 2:
        raise ValueError("Ожидается двумерная матрица")
    return [np.unique(row).tolist() for row in matrix]


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)
    if matrix.ndim != 2:
        raise ValueError("Ожидается двумерная матрица")
    return [np.unique(column).tolist() for column in matrix.T]


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    if data.rows <= 0 or data.columns <= 0:
        raise ValueError("Размеры матрицы должны быть положительными")
    generator = np.random.default_rng(data.seed)
    matrix = generator.normal(data.mean, data.std, size=(data.rows, data.columns))
    return MatrixStatistics(
        matrix=matrix,
        row_means=matrix.mean(axis=1),
        column_means=matrix.mean(axis=0),
        row_variances=matrix.var(axis=1),
        column_variances=matrix.var(axis=0),
    )


def chess(data: ChessInput) -> np.ndarray:
    row_indices, column_indices = np.indices((data.rows, data.columns))
    return np.where((row_indices + column_indices) % 2 == 0, data.first, data.second)


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    if data.width < 0 or data.height < 0:
        raise ValueError("Ширина и высота прямоугольника не могут быть отрицательными")
    image = np.empty((data.image_height, data.image_width, 3), dtype=np.uint8)
    image[:] = data.background_color
    left = (data.image_width - data.width) // 2
    top = (data.image_height - data.height) // 2
    # Обрезаем фигуру по границам изображения, если она больше холста.
    image[
        max(0, top):min(data.image_height, top + data.height),
        max(0, left):min(data.image_width, left + data.width),
    ] = data.shape_color
    return image


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    if data.semi_axis_x <= 0 or data.semi_axis_y <= 0:
        raise ValueError("Полуоси эллипса должны быть положительными")
    image = np.empty((data.image_height, data.image_width, 3), dtype=np.uint8)
    image[:] = data.background_color
    y, x = np.ogrid[:data.image_height, :data.image_width]
    center_x = (data.image_width - 1) / 2
    center_y = (data.image_height - 1) / 2
    # Убираем деление: округление дробей не должно исключать точки на границе.
    inside = (
        (x - center_x) ** 2 * data.semi_axis_y ** 2
        + (y - center_y) ** 2 * data.semi_axis_x ** 2
        <= (data.semi_axis_x * data.semi_axis_y) ** 2
    )
    image[inside] = data.shape_color
    return image


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values = np.asarray(data.values)
    window = data.window
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Временной ряд должен быть непустым одномерным массивом")
    if not 1 <= window <= values.size:
        raise ValueError("Окно должно быть от 1 до длины ряда")
    middle = values[1:-1]
    maxima = np.flatnonzero((middle > values[:-2]) & (middle > values[2:])) + 1
    minima = np.flatnonzero((middle < values[:-2]) & (middle < values[2:])) + 1
    # Сначала сравниваем исходные числа: float64 может объединить соседние
    # большие целые числа в одно значение и потерять строгий экстремум.
    numeric_values = values.astype(float)
    variance_values = numeric_values
    if np.issubdtype(values.dtype, np.integer):
        # Сдвиг не меняет дисперсию. Вычитаем точно до перехода к float,
        # используя Python int (object), чтобы избежать переполнения int64.
        variance_values = (values.astype(object) - int(values[0])).astype(float)
    return TimeSeriesStatistics(
        mean=float(numeric_values.mean()),
        variance=float(variance_values.var()),
        std=float(variance_values.std()),
        local_maxima_indices=maxima,
        local_minima_indices=minima,
        moving_average=np.convolve(numeric_values, np.ones(window) / window, mode="valid"),
    )


def one_hot(data: OneHotInput) -> np.ndarray:
    labels = np.asarray(data.labels)
    if labels.ndim != 1:
        raise ValueError("Метки должны быть одномерным вектором")
    if labels.size and (not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0)):
        raise ValueError("Метки должны быть неотрицательными целыми числами")
    required_count = int(labels.max()) + 1 if labels.size else 0
    class_count = required_count if data.class_count is None else data.class_count
    if class_count < required_count:
        raise ValueError("Количество классов недостаточно для заданных меток")
    result = np.zeros((labels.size, class_count), dtype=int)
    result[np.arange(labels.size), labels.astype(np.intp)] = 1
    return result
