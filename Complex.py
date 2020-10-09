from __future__ import annotations  # Should become unnecessary in Python 3.10
from typing import Tuple, Union, List

from sage.all import var, diff, Matrix, Rational, I, Expression

from .Riemannian import RiemChart
from .Base import Chart
from .Tensor import Tensor, Form


class CplxChart(Chart):
    """
    Class for charts on complex manifolds, uses (anti-)holomorphic coordinates

    :Attributes:
        * **self.dim** -- real dimension (inherited from Chart)

        * **self.coords** -- coordinate variables (inherited from Chart)

        * **self.cplx_dim** -- complex dimension = self.dim/2

        * **self.hol** -- holomorphic coordinates

        * **self.ahol** -- anti-holomorphic coordinates

    """

    def __init__(self, cplx_dim: int, coords: Tuple[list, list] = None):
        """
        :param cplx_dim: complex dimension of the manifold - becomes self.Cdim
        :param coords: tuple of (holomorphic, anti-holomorphic) coordinates, optional,
            defaults to ((z1, ..., zn), (z1b, ..., znb)) - becomes self.coords and self.hol, self.ahol
        :raises IndexError: if coords does not consist of 2 lists each of length cplx_dim

        Create a chart object for a complex manifold
        """
        self.cplx_dim = cplx_dim
        if coords:
            if len(coords) != 2:
                raise IndexError("Coordinates must be a pair (holomorphic, anti-holomorphic)!")
            elif len(coords[0]) != cplx_dim or len(coords[1]) != cplx_dim:
                raise IndexError(
                    "The number of holomorphic and anti-holomorphic coordinates must match the complex dimension!")
            self.hol = coords[0]
            self.ahol = coords[1]
        else:
            self.hol = [var(f'z{i + 1}') for i in range(self.cplx_dim)]
            self.ahol = [var(f'z{i + 1}b') for i in range(self.cplx_dim)]
        Chart.__init__(self, 2 * cplx_dim, self.hol + self.ahol)


class CplxForm(Form):
    """
    A class for forms on complex manifolds that have defined signature (holomorphic, anti-holomorphic)

    :Attributes:
        * **self.chart** -- The chart on which the form lives (inherited from Form)

        * **self.n** -- the total order of the form (inherited from Form)

        * **self.vals** -- the values of the form, stored as a dictionary with frozenset entries (inherited from Form)

        * **self.signature** -- signature of the form

    """

    def __init__(self, chart: CplxChart, signature: Tuple[int, int], vals: dict = None):
        """
        :param chart: the chart on which the form lives, becomes self.M
        :param signature: the signature of the form, becomes self.signature
        :param vals: the values of the form, becomes self.vals, optional, defaults to 0
        :raises IndexError: if the signature is not a pair
        :raises ValueError: if one (or both) of the components of signature is negative
        :raises NotImplementedError: if chart is not a CplxChart

        Create a Complex form object
        """
        if len(signature) != 2:
            raise IndexError("Signature must consist of 2 integers!")
        if signature[0] < 0 or signature[1] < 0:
            raise ValueError("Signature cannot be negative!")
        if not isinstance(chart, CplxChart):
            raise NotImplementedError('chart must be an instance of CplxChart class!')
        self.signature = signature
        Form.__init__(self, chart, sum(signature), vals)
        self.chart = chart

    def d_hol(self) -> CplxForm:
        """
        :return: d(self)

        The holomorphic exterior derivative
        """

        if self.signature[0] == self.chart.cplx_dim:
            return CplxForm(self.chart, (self.signature[0] + 1, self.signature[1]))

        vals = {}
        for ind in self.vals:
            for i in range(self.chart.cplx_dim):
                if i not in ind:
                    sgn = 1
                    for j in ind:
                        if j < i: sgn *= -1
                    ind_new = set(ind)
                    ind_new.add(i)
                    ind_new = frozenset(ind_new)
                    vals[ind_new] = vals.setdefault(ind_new, 0) + sgn * diff(self.vals[ind], self.chart.coords[i])

        for ind in list(vals.keys()):
            try:
                vals[ind] = vals[ind].simplify_full()
            except:  # noqa
                pass
            if vals[ind] == 0: vals.pop(ind)

        return CplxForm(self.chart, (self.signature[0] + 1, self.signature[1]), vals)

    def d_ahol(self) -> CplxForm:
        """
        :return: \\bar{d}(self)

        The anti-holomorphic exterior derivative
        """

        if self.signature[1] == self.chart.cplx_dim:
            return CplxForm(self.chart, (self.signature[0], self.signature[1] + 1))

        vals = {}
        for ind in self.vals:
            for i in range(self.chart.cplx_dim, self.chart.dim):
                if i not in ind:
                    sgn = 1
                    for j in ind:
                        if j < i: sgn *= -1
                    ind_new = set(ind)
                    ind_new.add(i)
                    ind_new = frozenset(ind_new)
                    vals[ind_new] = vals.setdefault(ind_new, 0) + sgn * diff(self.vals[ind], self.chart.coords[i])

        for ind in list(vals.keys()):
            try:
                vals[ind] = vals[ind].simplify_full()
            except:  # noqa
                pass
            if vals[ind] == 0: vals.pop(ind)

        return CplxForm(self.chart, (self.signature[0], self.signature[1] + 1), vals)


class HermMetric(Tensor):
    """
    Hermitian Metric class (h_{ab} dz^a (x) d\\bar{z}^b)

    :Attributes:
        * **self.chart** -- Chart to which this tensor belongs (inherited from Tensor)

        * **self.typ** -- a 2-tuple: (2,0) (inherited from Tensor)

        * **self.size** -- total size of the tensor: 2 (inherited from Tensor)

        * **self.vals** -- values of the tensor vals[i1][i2] (inherited from Tensor)

        * **self.h_vals** -- values of the tensor as coefficients of dz^a (x) d\\bar{z}^b

    """

    def __init__(self, chart: CplxChart, vals: list = None):
        """
        :param chart: Chart to which this tensor belongs, must be a CplxChart
        :param vals: Values as coefficients of dz^a (x) d\bar{z}^b, optional, defaults to \sum_a dz^a (x) d\bar{z}^a
            becomes self.h_vals

        Create a Hermitian Metric object
        """
        if not isinstance(chart, CplxChart):
            raise NotImplementedError('chart must be an instance of CplxChart class!')
        if vals:
            if len(vals) != chart.cplx_dim:
                raise IndexError("vals must be n x n where n is the complex dimension of the chart")
            for i in range(chart.cplx_dim):
                if len(vals[i]) != chart.cplx_dim:
                    raise IndexError("vals must be n x n where n is the complex dimension of the chart")
            self.h_vals = vals
        else:
            self.h_vals = [[1 if i == j else 0 for i in range(chart.cplx_dim)] for j in range(chart.cplx_dim)]

        Tensor.__init__(self, chart, chart.cplx_dim, 0,
                        [[self.h_vals[i][j - chart.cplx_dim] if i < chart.cplx_dim <= j else 0
                          for j in range(2 * chart.cplx_dim)] for i in range(2 * chart.cplx_dim)])
        self.chart = chart

    def __add__(self, other: Union[Tensor, HermMetric]) -> Union[Tensor, HermMetric]:
        """
        :param other: either a Hermitian metric, or another (2,0) tensor to be added
        :return: self + other (as Hermitian metric if other is one, otherwise as a (2,0) tensor)
        :raises NotImplementedError: if the two Hermitian metrics live on different charts.

        Add something to this Hermitian metric
        """
        if not isinstance(other, HermMetric):
            return Tensor.__add__(self, other)
        if self.chart != other.chart:
            raise NotImplementedError("Can't add Hermitian metrics that live on different charts")
        new_hv = [[self.h_vals[i][j] + other.h_vals[i][j] for j in range(self.chart.cplx_dim)] for i in
                  range(self.chart.cplx_dim)]
        return HermMetric(self.chart, new_hv)

    def __sub__(self, other: Union[Tensor, HermMetric]) -> Union[Tensor, HermMetric]:
        """
        :param other: either a Hermitian metric, or another (2,0) tensor to be subtracted
        :return: self - other (as Hermitian metric if other is one, otherwise as a (2,0) tensor)
        :raises NotImplementedError: if the two Hermitian metrics live on different charts.

        Add something to this Hermitian metric
        """
        if not isinstance(other, HermMetric):
            return Tensor.__sub__(self, other)
        if self.chart != other.chart:
            raise NotImplementedError("Can't subtract Hermitian metrics that live on different charts")
        new_hv = [[self.h_vals[i][j] - other.h_vals[i][j] for j in range(self.chart.cplx_dim)] for i in
                  range(self.chart.cplx_dim)]
        return HermMetric(self.chart, new_hv)

    def to_riem_metric(self) -> Matrix:
        """
        :return: The resulting Riemannian metric

        Create a (pseudo-)Riemannian metric from this Hermitian metric
        """
        g = Rational(1 / 2) * Matrix(self.vals)
        g += g.transpose()
        return g

    def to_form(self) -> CplxForm:
        """
        :return: The associated (1,1) form

        Create the 2-form associated to this Hermitian metric
        """
        om = CplxForm(self.chart, (1, 1))
        for i in range(self.chart.cplx_dim):
            for j in range(self.chart.cplx_dim):
                if self.h_vals[i][j]:
                    om.vals[frozenset({i, self.chart.cplx_dim + j})] = I * Rational(1 / 2) * self.h_vals[i][j]
        return om


class KahlerChart(RiemChart, CplxChart):
    """
    Class for handling Charts on Kähler manifolds

    Properties:
        * **self.dim** -- dimension (inherited from Chart)

        * **self.coords** -- coordinate variables (inherited from Chart)

        * **self.g** -- metric (inherited from RiemChart)

        * **self.ginv** -- inverse of the metric (inherited from RiemChart)

        * **self.CS** -- Christoffel symbols (computed by self.Compute_CS) (inherited from RiemChart)

        * **self.r** -- Curvature tensor as a list (computed by self.Compute_R) (inherited from RiemChart)

        * **self.r_t** -- Curvature tensor as a tensor object (computed by self.Compute_R) (inherited from RiemChart)

        * **self.ric_t** -- Weighted ricci tensor as a tensor object (computed by self.Compute_Ricci_T) (inherited from RiemChart)

        * **self.Cdim** -- complex dimension = self.n/2 (inherited from CplxChart)

        * **self.hol** -- holomorphic coordinates (inherited from CplxChart)

        * **self.ahol** -- anti-holomorphic coordinates (inherited from CplxChart)

        * **self.potential** -- Kähler potential function

        * **self.HM** -- Hermitian metric

        * **self.om** -- symplectic form

    """

    def __init__(self, coords: Tuple[list, list], potential: Expression = None):
        """
        :param coords: tuple of (holomorphic, anti-holomorphic) coordinates
        :param potential: the Kähler potential function, optional, defaults to (1/2)|z|^2 (Euclidean metric on C^n$
        :raises IndexError: if coords is not a 2-tuple where both components have the same length

        Create a chart for a Kähler manifold from a given Kähler potential
        """
        if len(coords) != 2:
            raise IndexError("Coordinates must be a pair (holomorphic, anti-holomorphic)!")
        elif len(coords[0]) != len(coords[1]):
            raise IndexError(
                "The number of holomorphic and anti-holomorphic coordinates must match the complex dimension!")
        if not potential:
            potential = Rational(1 / 2) * sum(coords[0][i] * coords[1][i] for i in range(len(coords[0])))
        self.potential = potential
        CplxChart.__init__(self, len(coords[0]), coords)
        self.herm_metric = HermMetric(self, [[diff(potential, self.ahol[j], self.hol[i]) for j in range(self.cplx_dim)]
                                             for i in range(self.cplx_dim)])
        self.g = self.herm_metric.to_riem_metric()
        self.ginv = self.g.inverse()
        self.om = self.herm_metric.to_form()

    def to_riem_chart(self, real_coords: List[tuple] = None) -> RiemChart:
        """
        :param real_coords: The names or real coordinates to be used, as a list of pairs (real_part, im_part), optional,
            defaults to [(x1, y1), (x2, y2), ..., (xn, yn)]
        :type real_coords: list of 2-tuples
        :return: Real version of this chart

        Convert this chart into a simple RiemChart object with real coordinates
        """

        def __met_gen(i, j):
            k = i // 2
            l = j // 2
            if (i - j) % 2 == 0:  # g(x_k, x_l) = g(y_k, y_l) = g(zk, zlb) + g(zl, zkb)
                return self.g[k][l + self.cplx_dim] + self.g[l][k + self.cplx_dim]
            elif i % 2:  # g(y_k, x_l) = I*( g(zk, zlb) - g(zl, zkb) )
                return I * (self.g[k][l + self.cplx_dim] - self.g[l][k + self.cplx_dim])
            else:  # g(x_k, y_l) = I*( g(zl, zkb) - g(zk, zlb) )
                return I * (self.g[l][k + self.cplx_dim] - self.g[k][l + self.cplx_dim])

        if real_coords:
            if len(real_coords) != self.cplx_dim:
                raise IndexError("real_coords must be a list of n 2-tuples")
            else:
                for i in range(self.cplx_dim):
                    if len(real_coords[i]) != 2:
                        raise IndexError("real_coords must be a list of n 2-tuples")

        if not real_coords:
            real_coords = [var(f'x{i + 1} y{i + 1}') for i in range(self.cplx_dim)]

        rc_list = []
        for pair in real_coords:
            rc_list += list(pair)

        subs = {**{self.hol[i]: real_coords[i][0] + I * real_coords[i][1] for i in range(self.cplx_dim)},
                **{self.ahol[i]: real_coords[i][0] - I * real_coords[i][1] for i in range(self.cplx_dim)}}
        # The above should be simplified from {**a, **b} to a|b in Python 3.9

        new_metric = Matrix([[__met_gen(i, j) for j in range(2 * self.cplx_dim)] for i in range(2 * self.cplx_dim)])
        new_metric = new_metric.substitute(subs)

        try:
            new_metric = new_metric.simplify_full()
        except:  # noqa
            pass

        return RiemChart(self.dim, rc_list, new_metric)
