from sage.all import Rational, Matrix

from .Tensor import Tensor
from .Riemannian import GenChart


class StatChart(GenChart):
    """
    A class for charts of statistical manifolds. Given a real parameter (α) and a symmetric (3,0)-tensor
    (Amari-Chentsov tensor D), the modified connection is the unique torsion-free connection satisfying:

    .. math::
        \\tilde{\\nabla} g = \\alpha D

    equivalently:

    .. math::
        g(\\tilde{\\nabla}_X Y, Z) = g(\\nabla_X Y, Z) - \\frac{\\alpha}{2}D(X,Y,Z)

    :Attributes:
        * **self.dim** -- dimension (inherited from Chart)

        * **self.coords** -- coordinate variables (inherited from Chart)

        * **self.g** -- metric (inherited from RiemChart)

        * **self.ginv** -- inverse of the metric (inherited from RiemChart)

        * **self.cs** -- Christoffel symbols (inherited from RiemChart)

        * **self.r** -- Curvature tensor as a list (inherited from RiemChart)

        * **self.r_t** -- Curvature tensor as a tensor object (inherited from RiemChart)

        * **self.ric_t** -- Weighted ricci tensor as a tensor object (inherited from RiemChart)

        * **self.mcs** -- Modified Christoffel symbols (inherited from GenChart) -- α-connection

        * **self.mr** -- Modified curvature tensor as a list (inherited from GenChart)

        * **self.mrt** -- Modified curvature tensor as a tensor object (inherited from GenChart)

        * **self.mric_t** -- Modified ricci tensor as a list (inherited from GenChart)

        * **self.alpha** -- The connection parameter (α)

        * **self.ac_tensor** -- The (3,0) Amari-Chentsov tensor

        * **self.ac_21** -- The (2,1) Amari-Chentsov tensor (auto-computed by self._compute_mcs)

    """

    def __init__(self, dim: int, coords: list = None, metric: Matrix = None, ac: list = None, alpha: float = 1):
        """
        :param dim: real dimension of the manifold - becomes self.n
        :param coords: names of the coordinate variables, optional, defaults to x1, x2, x3, ..., xn
        :param metric: the Riemannian metric, optional, defaults to dx1^2+dx2^2+...+dxn^2 - becomes a matrix self.g
        :param ac: the (3,0) Amari-Chentsov tensor (as a list), optional, defaults to all zeros
        :param alpha: the connection parameter, optional, defaults to 1
        :raises TypeError: if the provided Amari-Chentsov tensor is not symmetric

        Create a chart object of a statistical manifold
        """

        GenChart.__init__(self, dim, coords, metric)
        if not ac:
            ac = [[[0 for _ in range(dim)] for _ in range(dim)] for _ in range(dim)]
        self.ac_tensor = Tensor(self, 3, 0, ac)
        if not self.ac_tensor.is_input_symmetric():
            raise TypeError('Amari-Chentsov tensor must be symmetric')
        self.alpha = alpha
        self._compute_mcs()

    def _compute_mcs(self):
        """
        Compute the Christoffel Symbols of the \\alpha-connection in local coordinates.

        sets the correct values for self.mcs, with self.mcs[i][j][k] = \\Gamma^{(\\alpha)}_{ij}^k
        """

        self.ac_21 = self.ac_tensor.raise_ind(0)

        for i in range(self.dim):
            for j in range(self.dim):
                for k in range(self.dim):
                    self._mcs[i][j][k] -= Rational('1/2') * self.alpha * self.ac_21[(i, j, k)]

## Local Variables:
## mode: Python
## End:
