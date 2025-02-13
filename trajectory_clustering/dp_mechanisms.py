import math
import secrets
from typing import Callable, TypeVar
from opendp.measurements import make_laplace
from opendp.domains import vector_domain, atom_domain
from opendp.metrics import l1_distance
from opendp.mod import enable_features

import numpy as np

R = TypeVar("R")
T = TypeVar("T")

enable_features("contrib")


def choice(elements: list[R], probabilities: list[float]) -> R:
    assert abs(sum(probabilities) - 1.0) < 1e-9
    assert len(elements) == len(probabilities)

    cummulative = []
    current = 0
    for p in probabilities:
        current += p
        cummulative.append(current)

    rand_num = secrets.randbelow(10**9) / 10**9

    for i, cum in enumerate(cummulative):
        if rand_num < cum:
            return elements[i]

    raise ValueError("could not choose an element")


def random_int(a: int, b: int) -> int:
    """Return a random integer in the range (a, b]."""
    return secrets.randbelow(b - a) + a + 1


def exponential_mechanism(
    elements: list[R], u: Callable[[R], float], sensitivity: float, eps: float
) -> R:
    scores = [u(x) for x in elements]
    abs_probabilties = np.array(
        [math.exp(eps / (2 * sensitivity) * score) for score in scores]
    )
    normalization = sum(abs_probabilties)
    probabilities = abs_probabilties / normalization

    return choice(elements, probabilities.tolist())


def laplace_mechanism(
    x: R, f: Callable[[R], list[T]], sensitivity: float, eps: float
) -> list[T]:
    f_x = np.array(f(x))
    d_type = type(f_x[0])
    input_space = vector_domain(atom_domain(T=d_type)), l1_distance(T=d_type)
    noisy_result = np.array(make_laplace(*input_space, scale=sensitivity / eps)(f_x))
    return noisy_result.tolist()
