from typing import Tuple

from sage.all import Matrix, diff, Rational, var, function, Expression

from .Base import Chart


class RiemChart(Chart):
    """
    The class for coordinate charts on (pseudo-)Riemannian manifolds

    :Attributes:
        * **self.dim** -- The dimension of the manifold chart (inherited from Chart)

        * **self.coords** -- The variables being used as the coordinates on this chart (inherited from Chart)

        * **self.g** -- The (pseudo-)Riemannian metric stored as a Matrix object

        * **self.ginv** -- The inverse of the metric stored as a Matrix object

        * **self.cs** -- Christoffel symbols (auto-computed by self._compute_cs)

        * **self.r** -- Curvature tensor as a list (auto-computed by self._compute_r)

        * **self.r_t** -- Curvature tensor as a tensor object (auto-computed by self._compute_r)

        * **self.ric_t** -- Ricci tensor as a tensor object (auto-computed by self._compute_ricci_t)

    """

    def __init__(self, dim: int, coords: list=None, metric: Matrix=None):
        """
        :param dim: real dimension of the manifold - becomes self.n
        :param coords: names of the coordinate variables, optional, defaults to x1, x2, x3, ..., xn
        :param metric: the Riemannian metric, optional, defaults to dx1^2+dx2^2+...+dxn^2 - becomes a matrix self.g
        :raises ValueError: if sage fails to convert the metric to a Matrix
        :raises IndexError: if the metric is not dim x dim

        Create a Riemannian manifold chart object
        """
        Chart.__init__(self, dim, coords)
        if metric:
            try:
                self.g = Matrix(metric)
            except ValueError:
                raise ValueError('Could not convert the provided metric to a Matrix')
            if self.g.dimensions() != (dim, dim):
                raise IndexError('The provided metric is incorrect size')
        else:
            # noinspection PyUnresolvedReferences
            self.g = Matrix.identity(dim)
        self.ginv = self.g.inverse()

    def inner(self, v1: list, v2: list) -> Expression:
        """
        :param v1: First vector (field)
        :param v2: Second vector (field)
        :return: `g(v1,v2)`
        :raises IndexError: when one or both of the vectors are wrong length

        Compute the inner product of two vectors (vector fields)
        """

        if len(v1) != self.dim or len(v2) != self.dim:
            raise IndexError('Vectors must have length = dimension')
        return sum([v2[j] * sum([v1[i] * self.g[i][j] for i in range(self.dim)]) for j in range(self.dim)])

    @property
    def cs(self):
        if not hasattr(self, '_CS'):
            self._compute_cs()
        return self._CS

    def _compute_cs(self):
        """
        Compute the Christoffel Symbols in local coordinates.

        Creates self._CS, with self._CS[i][j][k] = \\Gamma_{ij}^k
        """
        self._CS = [[[(Rational('1/2') * sum([(diff(self.g[j][k], self.coords[i])
                                               + diff(self.g[k][i], self.coords[j])
                                               - diff(self.g[i][j], self.coords[k])) * self.ginv[k][m]
                                              for k in range(self.dim)])).simplify() for m in range(self.dim)]
                     for j in range(self.dim)] for i in range(self.dim)]

    def connection(self, vf1: list, vf2: list) -> list:
        """
        :param vf1: First vector field - vector field along which we are taking the covariant derivative
        :param vf2: Second vector field - vector field being differentiated
        :return: The covariant derivative of `vf2` along `vf1`
        :raises IndexError: if one (or both) of the vector fields has the wrong size

        Compute the covariant derivative of `vf2` along `vf1`

        .. math:: \\nabla_{\\text{vf1}} \\text{vf2}
        """

        if len(vf1) != self.dim or len(vf2) != self.dim:
            raise IndexError('Vector fields must have length = dimension')

        p1 = [sum([vf1[i] * diff(vf2[j], self.coords[i]) for i in range(self.dim)]) for j in range(self.dim)]
        p2 = [sum([vf2[j] * sum([vf1[i] * self.cs[i][j][m] for i in range(self.dim)]) for j in range(self.dim)])
              for m in range(self.dim)]

        return [p1[i] + p2[i] for i in range(self.dim)]

    @property
    def r(self):
        if not hasattr(self, '_R'):
            self._compute_r()
        return self._R

    @property
    def r_t(self):
        if not hasattr(self, '_R_T'):
            self._compute_r()
        return self._R_T

    def _compute_r(self):
        """
        Compute the curvature in local coordinates. Creates two objects:

        self._R - list version self.r[i][j][k][l] = R_{ijk}^l
        self._R_T - Tensor version
        """
        from .Tensor import Tensor

        self._R = [[[[sum([self.cs[j][k][l] * self.cs[i][l][s] - self.cs[i][k][l] * self.cs[j][l][s]
                           for l in range(self.dim)]) + diff(self.cs[j][k][s], self.coords[i])
                      - diff(self.cs[i][k][s], self.coords[j]) for s in range(self.dim)] for k in range(self.dim)]
                    for j in range(self.dim)] for i in range(self.dim)]

        self._R_T = Tensor(self, 3, 1, self.r)

    def curv(self, v1: list, v2: list, v3: list) -> list:
        """
        :param v1: First vector (field)
        :param v2: Second vector (field)
        :param v3: Third vector (field)
        :return: Curvature `R(v1,v2)v3`
        :raises IndexError: if one (or more) of the vector fields has the wrong size

        Compute the curvature tensor on the three input vectors (vector fields)
        """

        if len(v1) != self.dim or len(v2) != self.dim or len(v3) != self.dim:
            raise IndexError('Vector fields must have length = dimension')
        return [
            sum([v3[k] * sum([v2[j] * sum([v1[i] * self.r[i][j][k][m] for i in range(self.dim)]) for j in range(self.dim)])
                 for k in range(self.dim)]) for m in range(self.dim)]

    def sec(self, v1: list, v2: list) -> Expression:
        """
        :param v1: First vector (field)
        :param v2: Second vector (field)
        :return: Sectional curvature of the plane(s) spanned by `v1` and `v2`
        :raises IndexError: if one (or both) of the vector fields has the wrong size

        Compute the sectional curvature of the plane spanned by the given vectors (vector fields)
        """

        if len(v1) != self.dim or len(v2) != self.dim:
            raise IndexError('Vector fields must have length = dimension')
        return self.inner(self.curv(v1, v2, v2), v1) / (
                self.inner(v1, v1) * self.inner(v2, v2) - self.inner(v1, v2) ** 2)

    def ricci(self, v1: list, v2: list) -> Expression:
        """
        :param v1: First vector (field)
        :param v2: Second vector (field)
        :return: Ricci curvature Ric(v1,v2)
        :raises IndexError: if one (or both) of the vector fields has the wrong size

        Compute the Ricci curvature on the two given vectors (vector fields)
        """

        if len(v1) != self.dim or len(v2) != self.dim:
            raise IndexError('Vector fields must have length = dimension')
        basis = [[1 if i == j else 0 for i in range(self.dim)] for j in range(self.dim)]

        res = 0

        for i in range(self.dim):
            res += self.curv(basis[i], v1, v2)[i]

        return res

    @property
    def ric_t(self):
        if not hasattr(self, '_Ric_T'):
            self._compute_ricci_t()
        return self._Ric_T

    def _compute_ricci_t(self):
        """
        Compute the Ricci curvature tensor as a tensor object

        Creates tensor object self.ric_t
        """

        self._Ric_T = self.r_t.iotrace(0, 0)

    def path_derivative(self, path: list, vf: list) -> list:
        """
        :param path: Path along which to differentiate - a parametric curve on this Chart, with parameter = t
        :param vf: Vector field to be differentiated - using the same parametrization
        :return: The path derivative of `vf` along `gamma` as a parametrized vector field
        :raises IndexError: if the vector field or path is the wrong size

        Compute the path derivative of a vector field along a path

        .. math:: \\nabla_{\\dot{\\gamma}} \\text{vf}
        """

        if len(path) != self.dim or len(vf) != self.dim:
            raise IndexError('Path and vector field must have length = dimension')

        t = var('t')
        direction = [diff(path[i], t) for i in range(self.dim)]
        dxn = [[1 if i == j else 0 for i in range(self.dim)] for j in range(self.dim)]
        res = [diff(vf[i], t) for i in range(self.dim)]
        d = {self.coords[i]: path[i] for i in range(self.dim)}
        for i in range(self.dim):
            tmp = self.connection(direction, dxn[i])
            res = [res[j] + vf[i] * tmp[j].subs(d) for j in range(self.dim)]
        return res

    def parallel_transport(self, path: list) -> Tuple[list, list]:
        """
        :param path: Path along which to differentiate - a parametric curve along
        :return: (`vfs`, `des`) (see below)
        :raises IndexError: if the path is wrong size

        * `vfs` = names of the components of the vector field,
        * `des` = system of differential equations

        Set up a system of ODEs to determine parallel transport along the given path
        """

        if len(path) != self.dim:
            raise IndexError('Path must have length = dimension')
        t = var('t')
        vfs = [function('f' + str(i + 1))(t) for i in range(self.dim)]
        pd = self.path_derivative(path, vfs)
        des = [pd[i] == 0 for i in range(self.dim)]
        return vfs, des


class GenChart(RiemChart):
    """
    Dummy class for coordinate charts on (pseudo-)Riemannian manifolds with a modified connection. This class
    should not be used directly, rather create a subclass that implements the structure you are interested in.

    :Attributes:
        * **self.dim** -- The dimension of the manifold chart (inherited from Chart)

        * **self.coords** -- The variables being used as the coordinates on this chart (inherited from Chart)

        * **self.g** -- The (pseudo-)Riemannian metric stored as a Matrix object (inherited from RiemChart)

        * **self.ginv** -- The inverse of the metric stored as a Matrix object (inherited from RiemChart)

        * **self.cs** -- Christoffel symbols (auto-computed by self._compute_cs) (inherited from RiemChart)

        * **self.r** -- Curvature tensor as a list (auto-computed by self._compute_r) (inherited from RiemChart)

        * **self.r_t** -- Curvature tensor as a tensor object (auto-computed by self._compute_r) (inherited from RiemChart)

        * **self.ric_t** -- Ricci tensor as a tensor object (auto-computed by self._compute_ricci_t) (inherited from RiemChart)

        * **self.mcs** -- Modified Christoffel symbols

        * **self.mr** -- Modified curvature tensor as a list (auto-computed by self._compute_mr)

        * **self.mr_t** -- Modified curvature tensor as a tensor object (auto-computed by self._compute_mr)

        * **self.mric_t** -- Modified Ricci tensor as a tensor object (auto-computed by self._compute_mricci_t)

    """

    def __init__(self, dim: int, coords: list=None, metric: Matrix=None, mcs: list=None):
        """
        :param dim: real dimension of the manifold - becomes self.n
        :param coords: names of the coordinate variables, optional, defaults to x1, x2, x3, ..., xn
        :param metric: the Riemannian metric, optional, defaults to dx1^2+dx2^2+...+dxn^2 - becomes a matrix self.g
        :param mcs: the modified connection, optional, defaults to the Levi-Civita connection - becomes self.mcs
        :raises IndexError: if the provided modified connection has size other than dim x dim x dim

        Create a Generalized Riemannian manifold chart object
        """

        RiemChart.__init__(self, dim, coords, metric)
        if mcs:
            if len(mcs) != self.dim:
                raise IndexError('mcs must be a 3-level nested list dim x dim x dim')
            else:
                for lvl in mcs:
                    if len(lvl) != self.dim:
                        raise IndexError('mcs must be a 3-level nested list dim x dim x dim')
                    else:
                        for sub_lvl in lvl:
                            if len(sub_lvl) != self.dim:
                                raise IndexError('mcs must be a 3-level nested list dim x dim x dim')
            self._mcs = mcs
        else:
            self._compute_cs()
            self._mcs = [[[self.cs[i][j][k] for k in range(dim)] for j in range(dim)] for i in range(dim)]

    def connection(self, vf1: list, vf2: list) -> list:
        """
        :param vf1: First vector field - vector field along which we are taking the covariant derivative
        :param vf2: Second vector field - vector field being differentiated
        :return: The covariant derivative of `vf2` along `vf1`
        :raises IndexError: if one (or both) of the vector fields has the wrong size

        Compute the covariant derivative of vf2 along vf1 (using self.mcs)
        overrides the corresponding method from RiemChart

        .. math:: \\tilde{\\nabla}_{\\text{vf1}} \\text{vf2}
        """

        if len(vf1) != self.dim or len(vf2) != self.dim:
            raise IndexError('Vector fields must have length = dimension')

        p1 = [sum([vf1[i] * diff(vf2[j], self.coords[i]) for i in range(self.dim)]) for j in range(self.dim)]
        p2 = [sum([vf2[j] * sum([vf1[i] * self.mcs[i][j][m] for i in range(self.dim)]) for j in range(self.dim)])
              for m in range(self.dim)]

        return [p1[i] + p2[i] for i in range(self.dim)]

    @property
    def mcs(self):
        return self._mcs

    @property
    def mr(self):
        if not hasattr(self, '_mr'):
            self._compute_mr()
        return self._mr

    @property
    def mr_t(self):
        """
        Test

        :return: Test
        """
        if not hasattr(self, '_mr_t'):
            self._compute_mr()
        return self._mr_t

    def _compute_mr(self):
        """
        Compute the curvature of self.mcs in local coordinates. Creates two objects:

        self.mr - list version self.mr[i][j][k][l] = MR_{ijk}^l
        self.mr_t - Tensor version
        """
        from .Tensor import Tensor

        self._mr = [[[[sum([self.mcs[j][k][l] * self.mcs[i][l][s] - self.mcs[i][k][l] * self.mcs[j][l][s]
                            for l in range(self.dim)]) + diff(self.mcs[j][k][s], self.coords[i])
                       - diff(self.mcs[i][k][s], self.coords[j]) for s in range(self.dim)] for k in range(self.dim)]
                     for j in range(self.dim)] for i in range(self.dim)]

        self._mr_t = Tensor(self, 3, 1, self.mr)

    @property
    def mric_t(self):
        if not hasattr(self, '_mric_t'):
            self._compute_mricci_t()
        return self._mric_t

    def _compute_mricci_t(self):
        """
        Compute the ricci curvature tensor of self.mcs as a tensor object

        Creates tensor object self.mric_t
        """

        self._mric_t = self.mr_t.iotrace(0, 0)

## Local Variables:
## mode: Python
## End:
