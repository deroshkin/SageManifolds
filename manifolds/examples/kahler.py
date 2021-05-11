from sage.all import var, log

from ..complex import KahlerChart


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