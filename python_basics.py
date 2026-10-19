"""Решения задач на базовый Python."""

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput


def count_vowels(data: TextInput) -> int:
    return sum(character in "aeiou" for character in data.value.lower())


def has_unique_characters(data: TextInput) -> bool:
    return len(set(data.value)) == len(data.value)


def count_one_bits(data: PositiveIntegerInput) -> int:
    if data.value < 0:
        raise ValueError("Число должно быть неотрицательным")
    return data.value.bit_count()


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    if number < 0:
        raise ValueError("Число должно быть неотрицательным")
    steps = 0
    while number >= 10:
        product = 1
        for digit in str(number):
            product *= int(digit)
        number = product
        steps += 1
    return steps


def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    if len(predicted) != len(expected) or len(predicted) == 0:
        raise ValueError("Векторы должны быть непустыми и одинаковой длины")
    # int предотвращает переполнение, если элементы — целые числа NumPy.
    return sum((int(x) - int(y)) ** 2 for x, y in zip(predicted, expected)) / len(predicted)


def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    if number < 1:
        raise ValueError("Число должно быть положительным")
    factors = []
    divisor = 2
    while divisor * divisor <= number:
        power = 0
        while number % divisor == 0:
            number //= divisor
            power += 1
        if power:
            factors.append(f"({divisor})" if power == 1 else f"({divisor}**{power})")
        divisor = 3 if divisor == 2 else divisor + 2
    if number > 1:
        factors.append(f"({number})")
    # У единицы нет простых множителей: результат — пустая строка.
    return "".join(factors)


def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value
    if cube_count < 1:
        raise ValueError("Количество кубиков должно быть положительным")
    # Сумма квадратов строго возрастает, поэтому можно искать двоичным поиском.
    left, right = 1, cube_count
    while left <= right:
        height = (left + right) // 2
        total = height * (height + 1) * (2 * height + 1) // 6
        if total == cube_count:
            return height
        if total < cube_count:
            left = height + 1
        else:
            right = height - 1
    return "It is impossible"


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    if data.value < 0:
        raise ValueError("Число должно быть неотрицательным")
    digits = str(data.value)
    side_length = (len(digits) - 1) // 2
    left = digits[:side_length]
    right = digits[len(digits) - side_length:]
    return sum(map(int, left)) == sum(map(int, right))
