from sage.all import diff, Matrix, Expression

from .riemannian import GenChart


class DensChart(GenChart):
    """
    A class for manifold with density charts (a la Bakry-Emery with N=1). This is done in the style of Wylie-Yeroshkin:

    .. math::
        \\varphi &= \\frac{f}{n-1}\\\\
        \\tilde{\\nabla}_X Y &= \\nabla_X Y - d\\varphi(X) Y - d\\varphi(Y) X

    :Attributes:
        * **self.dim** -- dimension (inherited from Chart)

        * **self.coords** -- coordinate variables (inherited from Chart)

        * **self.g** -- metric (inherited from RiemChart)

        * **self.ginv** -- inverse of the metric (inherited from RiemChart)

        * **self.CS** -- Christoffel symbols (inherited from RiemChart)

        * **self.r** -- Curvature tensor as a list (inherited from RiemChart)

        * **self.r_t** -- Curvature tensor as a tensor object (inherited from RiemChart)

        * **self.ric_t** -- Weighted ricci tensor as a tensor object (inherited from RiemChart)

        * **self.mcs** -- Modified Christoffel symbols (inherited from GenChart). Uses the Weighted connection from Wylie-Yeroshkin

        * **self.mr** -- Modified curvature tensor as a list (inherited from GenChart)

        * **self.mrt** -- Modified curvature tensor as a tensor object (inherited from GenChart)

        * **self.mric_t** -- Modified ricci tensor as a list (inherited from GenChart)

        * **self.phi** -- density function (normalization of phi = f/(n-1) from standard Bakry-Emery construction)

    """

    def __init__(self, dim: int, coords: list=None, metric: Matrix=None, phi: Expression=0):
        """
        :param dim: real dimension of the manifold - becomes self.dim
        :param coords: names of the coordinate variables, optional, defaults to x1, x2, x3, ..., xn
        :param metric: the Riemannian metric, optional, defaults to dx1^2+dx2^2+...+dxn^2 - becomes a matrix self.g
        :param phi: density function, optional, defaults to 0

        Create a manifold with density chart object
        """
        self.phi = phi
        GenChart.__init__(self, dim, coords, metric)

        self._compute_mcs()

    def _compute_mcs(self):
        """
        Compute the weighted Christoffel Symbols in local coordinates.

        sets the correct values for self.mcs, with self.mcs[i][j][k] = \\tilde{\\Gamma}_{ij}^k
        """

        for i in range(self.dim):
            self._mcs[i][i][i] -= 2 * diff(self.phi, self.coords[i])
            for j in range(i + 1, self.dim):
                self._mcs[i][j][i] -= diff(self.phi, self.coords[j])
                self._mcs[i][j][j] -= diff(self.phi, self.coords[i])
                self._mcs[j][i][i] -= diff(self.phi, self.coords[j])
                self._mcs[j][i][j] -= diff(self.phi, self.coords[i])

## Local Variables:
## mode: Python
## End:
