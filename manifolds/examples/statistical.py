from sage.all import var, diff

from ..statistical import StatChart


def discrete_stat_manifold(n: int, variables: list = None, alpha: float = 1) -> StatChart:
    """
    :param n: Number of distinct point
    :param variables: The names of the variables indicating the weights of each point except point 0, optional,
        the default values are [x1, ..., x{n-1}]
    :param alpha: The alpha parameter for the statistical manifold
    :return: The resulting (n-1)-dimensional Statistical Manifold object, well-defined on xi >= 0, sum xi <= 1

    Construct a statistical manifold for the full space of probability measures on `n` points.
    """

    if variables:
        if len(variables) != n-1:
            raise IndexError('The number of variables must be one less than the number of points.')
    else:
        variables = [var(f'x{i + 1}') for i in range(n - 1)]

    # probability distribution
    p = [1 - sum(variables)] + variables

    # Fisher metric
    fm = [[sum(diff(p[i], v1) * diff(p[i], v2) / p[i] for i in range(n)) for v1 in variables] for v2 in variables]
    # Amari-Chentsov symmetric 3-tensor
    ac = [[[sum(diff(p[i], v1) * diff(p[i], v2) * diff(p[i], v3) / p[i] ** 2 for i in range(n)) for v1 in variables]
           for v2 in variables] for v3 in variables]

    return StatChart(n-1, variables, fm, ac, alpha)