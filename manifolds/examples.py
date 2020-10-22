from sage.all import var, sin, Matrix, log

from .riemannian import RiemChart
from .complex import KahlerChart


def sphere_polar(n: int, variables: list=None) -> RiemChart:
    """
    :param n: dimension
    :param variables: Coordinates to use, optional, defaults to r, t1, t2, ... t{n-1}
    :return: a Riemmanian chart for a round sphere in polar coordinates
    :raises IndexError: if the provided list of variables has wrong length

    Create a Riemannian Chart for an n-dimensional radius 1 sphere in polar coordinates,
    misses a codimension 2 totally geodesic sphere
    """

    if variables:
        if len(variables) != n:
            raise IndexError('Number of variables is not the same as the dimension')
    else:
        var_names = ['r'] + [f't{i}' for i in range(1, n)]
        variables = var(' '.join(var_names))

    metric = [[0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        gii = 1
        for j in range(i):
            gii *= sin(variables[j]) ** 2
        metric[i][i] = gii

    return RiemChart(n, variables, metric)


def sphere_conformal(n: int, variables: list=None) -> RiemChart:
    """
    :param n: dimension
    :param variables: Coordinates to use, optional, defaults to x1, x2, ...
    :return: a Riemannian chart for a unit sphere in conformal coordinates
    :raises IndexError: if the provided list of variables has wrong length

    Create a Riemannian Chart for an n-dimensional radius 1 sphere in conformal coordinates,
    misses one point on the sphere
    """

    if variables:
        if len(variables) != n:
            raise IndexError('Number of variables is not the same as the dimension')
    else:
        var_names = [f'x{i + 1}' for i in range(n)]
        variables = var(' '.join(var_names))

    # noinspection PyUnresolvedReferences
    metric = Matrix.identity(n) * (4 / (1 + sum(v ** 2 for v in variables)) ** 2)

    return RiemChart(n, variables, metric)


def complex_projective_space(n: int, variables: list=None) -> KahlerChart:
    """
    :param n: (complex) dimension
    :param variables: (holomorphic, anti-holomorphic) coordinates to use, optional,
        defaults to [(z1, z2, ...), (z1b, z2b, ...)]
    :return: :py:class:`Manifolds.Complex.KahlerChart` object for the Fubini-Study complex projective space
    :raises IndexError: when the variables are not provided as a pair of lists of length n

    Create a Kähler Chart object for an n-dimensional complex projective space, using the Fubini-Study metric
    """

    if variables:
        if len(variables) != 2:
            raise IndexError('Variables must be a tuple (hol,ahol)')
        if len(variables[0]) != n or len(variables[1]) != n:
            raise IndexError('Wrong number of holomorphic or anti-holomorphic variables')
    else:
        hol_var_names = [f'z{i + 1}' for i in range(n)]
        ahol_var_names = [f'z{i + 1}b' for i in range(n)]
        variables = (var(' '.join(hol_var_names)), var(' '.join(ahol_var_names)))

    pot = log(1 + sum(variables[0][i] * variables[1][i] for i in range(n)))

    return KahlerChart(variables, pot)


def hyperbolic_half_space(n: int, variables: list=None) -> RiemChart:
    """
    :param n: dimension
    :param variables: Coordinates to use, optional, defaults to x1, x2, ...
    :return: a Riemannian chart for the upper half-space model of a hyperbolic space
    :raises IndexError: if the provided list of variables has wrong length

    Create a Riemannian Chart for an n-dimensional hyperbolic space with sectional curvature -1,
    uses the half-space model, with the last coordinate positive.
    """
    if variables:
        if len(variables) != n:
            raise IndexError('Number of variables is not the same as the dimension')
    else:
        var_names = [f'x{i + 1}' for i in range(n)]
        variables = var(' '.join(var_names))

    # noinspection PyUnresolvedReferences
    metric = Matrix.identity(n) * (1 / variables[-1] ** 2)

    return RiemChart(n, variables, metric)
