from __future__ import annotations # Should become unnecessary in Python 3.10
from typing import Union
import itertools

from sage.all import diff, Matrix, factorial, det, sqrt, prod, Expression

from .Riemannian import RiemChart, GenChart
from .Base import Chart


def sign(l):
    """
    :param l: ordered list of indices
    :type l: list (or similar)
    :return: sign of l

    Compute the sign of an ordered list (-1 for each flip needed to get to ascending order)
    """
    res = 1
    for i in range(len(l)):
        for j in range(i + 1, len(l)):
            if l[i] > l[j]: res *= -1
    return res


class Tensor:
    """
    Class for Tensor objects on manifold charts.

    You can add and subtract `Tensor` objects provided they are of the same time (#inputs, #outputs).

    You can also multiply two `Tensor` objects, or a `Tensor` object by an expression.
    Multiplying two `Tensor` objects is interpreted as a tensor project,
    the order of indices is 1st `Tensor` inputs, 2nd `Tensor` inputs, 1st `Tensor` outputs, 2nd `Tensor` outputs

    :Attributes:
        * **self.chart** -- Chart to which this tensor belongs

        * **self.typ** -- a 2-tuple: (#inputs, #outputs)

        * **self.size** -- total size of the tensor: #inputs+#outputs

        * **self.vals** -- values of the tensor vals[i1]..[im][o1]..[on]

    """

    def __init__(self, chart: Chart, n_in: int, n_out: int, vals: list=None):
        """
        :param chart: The manifold coordinate chart to which this tensor belongs
        :param n_in: Number of input vectors
        :param n_out: Number of output vectors
        :param vals: Values of the tensor vals[i1][i2]...[im][o1]...[on], optional, defaults to all zeros

        Create a Tensor object
        """
        if not isinstance(chart, Chart):
            raise NotImplementedError('chart must be an instance of Chart class!')
        self.chart = chart
        self.typ = (n_in, n_out)
        self.size = n_in + n_out

        if vals:
            self.vals = vals
        else:
            def __builder(lvls, size):
                if lvls == 1:
                    return [0] * size
                else:
                    return [__builder(lvls - 1, size) for _ in range(size)]

            self.vals = __builder(self.size, self.chart.dim)

    def __getitem__(self, ind: tuple) -> Union[Expression, list]:
        """
        :param ind: the index at which to look
        :return: the value or values at ind
        :raises IndexError: when length of ind is larger than total size of Tensor

        Get the value stored at ind, if length of ind < size of the tensor, returns the trailing sub-tensor
        """
        if len(ind) > self.size:
            raise IndexError
        else:
            l = self.vals
            for i in ind:
                l = l[i]
            return l

    def __setitem__(self, ind: tuple, val: Expression):
        """
        :param ind: index at which to set the value
        :type ind: list
        :param val: new value
        :type val: Expression (or similar)
        :return: val on success, None on failure
        :raises IndexError: if the length of `ind` is not the same as the total size of the Tensor

        Set the value of one entry of the tensor
        """
        if len(ind) != self.size:
            raise IndexError
        else:
            l = self.vals
            for i in ind[:-1]:
                l = l[i]
            l[ind[-1]] = val
            return val

    def __mul__(self, other: Union[Tensor, Expression]) -> Tensor:
        """
        :param other: the other tensor or function
        :return: the (tensor) product of self with other
        :raises TypeError: if the other `Tensor` is on a different `Chart`

        Compute the tensor product of 2 tensors (self (x) other) or multiply the tensor by a function
        """
        if isinstance(other, Tensor):
            if self.chart != other.chart:
                raise TypeError("Can't multiply tensors that live on different charts")
            s_in = self.typ[0]
            o_in = other.typ[0]
            s_out = self.typ[1]
            o_out = other.typ[1]
            n_in = s_in + o_in
            n_out = s_out + o_out

            res = Tensor(self.chart, n_in, n_out)

            for ind in itertools.product(*[range(self.chart.dim) for _ in range(n_in + n_out)]):
                s_ind = ind[:s_in] + ind[n_in:n_in + s_out]
                o_ind = ind[s_in:n_in] + ind[n_in + s_out:]
                # noinspection PyTypeChecker
                res[ind] = self[s_ind] * other[o_ind]

            return res
        else:
            def __builder(lvls, size, _ind):
                if lvls == 1:
                    return [self[_ind + [i]] * other for i in range(size)]
                else:
                    return [__builder(lvls - 1, size, _ind + [i]) for i in range(size)]

            vals = __builder(self.size, self.chart.dim, [])
            return Tensor(self.chart, self.typ[0], self.typ[1], vals)

    def __add__(self, other: Tensor) -> Tensor:
        """
        :param other: second tensor
        :return: self+other
        :raises TypeError: if impossible (different types / different `Chart`s)

        Add two tensors
        """
        if not isinstance(other, Tensor):
            raise TypeError("Can't add a non-tensor object to a tensor")
        if self.chart != other.chart:
            raise TypeError("Can't add tensors that live on different charts")
        if self.typ[0] != other.typ[0]:
            raise TypeError("Can't add tensors of differing types")
        if self.typ[1] != other.typ[1]:
            raise TypeError("Can't add tensors of differing types")

        def __builder(lvls, size, ind):
            if lvls == 1:
                return [self[ind + [i]] + other[ind + [i]] for i in range(size)]
            else:
                return [__builder(lvls - 1, size, ind + [i]) for i in range(size)]

        vals = __builder(self.size, self.chart.dim, [])
        return Tensor(self.chart, self.typ[0], self.typ[1], vals)

    def __sub__(self, other: Tensor) -> Tensor:
        """
        :param other: second tensor
        :return: self-other
        :raises NotImplementedError: if impossible (different types / different `Chart`s)

        Subtract two tensors
        """
        if not isinstance(other, Tensor):
            raise NotImplementedError("Can't subtract a non-tensor object from a tensor")
        if self.chart != other.chart:
            raise NotImplementedError("Can't subtract tensors that live on different charts")
        if self.typ[0] != other.typ[0]:
            raise NotImplementedError("Can't subtract tensors of differing types")
        if self.typ[1] != other.typ[1]:
            raise NotImplementedError("Can't subtract tensors of differing types")

        def __builder(lvls, size, ind):
            if lvls == 1:
                return [self[ind + [i]] - other[ind + [i]] for i in range(size)]
            else:
                return [__builder(lvls - 1, size, ind + [i]) for i in range(size)]

        vals = __builder(self.size, self.chart.dim, [])
        return Tensor(self.chart, self.typ[0], self.typ[1], vals)

    def raise_ind(self, ind: int) -> Tensor:
        """
        :param ind: index to be raised
        :return: modified tensor, raised index becomes the last output index
        :raises NotImplementedError: if no metric
        :raises IndexError: when trying to raise index outside the valid range

        Raise the given index, requires self.M to be a Riemannian chart
        """
        if ind >= self.typ[0] or ind < -self.typ[0]:
            raise IndexError('Index out of Range')
        if not isinstance(self.chart, RiemChart):
            raise NotImplementedError("Can't raise an index without a metric")

        ginv = Tensor(self.chart, 0, 2, self.chart.ginv)
        tmp = self * ginv
        return tmp.iotrace(ind, tmp.typ[1] - 1)

    def lower_ind(self, ind: int) -> Tensor:
        """
        :param ind: index to be lowered
        :return: modified tensor, lowered index becomes the last input index
        :raises NotImplementedError: if no metric
        :raises IndexError: when trying to raise index outside the valid range

        Lower the given index, requires self.M to be a Riemannian chart
        """
        if ind >= self.typ[1] or ind < -self.typ[1]:
            raise IndexError('Index out of Range')
        if not isinstance(self.chart, RiemChart):
            raise NotImplementedError("Can't lower an index without a metric")

        g = Tensor(self.chart, 2, 0, self.chart.g)
        tmp = self * g
        return tmp.iotrace(tmp.typ[0] - 1, ind)

    def diff(self) -> Tensor:
        """
        :return: The covariant derivative of the Tensor, the direction of differentiation is the new 0th input index
        :raises NotImplementedError: when the Chart does not have a connection associated with it

        Compute the covariant derivative of this tensor
        """

        if isinstance(self.chart, GenChart):
            cs = self.chart.mcs
        elif isinstance(self.chart, RiemChart):
            cs = self.chart.cs
        else:
            raise NotImplementedError("Can't compute a covariant derivative without a connection")

        d = Tensor(self.chart, self.typ[0] + 1, self.typ[1])

        for ind in itertools.product(*[range(self.chart.dim) for _ in range(self.size + 1)]):
            inp = ind[1:1 + self.typ[0]]
            outp = ind[-self.typ[1]:] if self.typ[1] else ()
            val = diff(self[ind[1:]], self.chart.coords[ind[0]])

            for t in range(self.typ[0]):
                inpt = list(inp)
                ut = inpt[t]
                for l in range(self.chart.dim):
                    inpt[t] = l
                    val -= self[tuple(inpt) + outp] * cs[ut][ind[0]][l]

            for t in range(self.typ[1]):
                outpt = list(outp)
                vt = outpt[t]
                for l in range(self.chart.dim):
                    outpt[t] = l
                    val += self[inp + tuple(outpt)] * cs[l][ind[0]][vt]

            try:
                d[ind] = val.simplify_full()
            except:  # noqa
                d[ind] = val
        return d

    def is_input_symmetric(self) -> bool:
        """
        :return: True if symmetric, otherwise False

        Verify whether the tensor is symmetric in its inputs
        """
        if self.typ[0] <= 1: return True
        for ind in itertools.combinations_with_replacement(range(self.chart.dim), self.typ[0]):
            base = self[ind]
            for i2 in itertools.permutations(ind, self.typ[0]):
                if base != self[i2]:
                    return False
        return True

    def is_codazzi(self) -> bool:
        """
        :return: True if Codazzi, otherwise False
        :raises TypeError: if trying to run on a tensor that is not (2,0)

        Check whether the given tensor is Codazzi
        """

        if self.typ != (2, 0):
            raise TypeError("Codazzi only makes sense for (2,0) tensors")
        if not self.is_input_symmetric():
            return False
        d = self.diff()
        return d.is_input_symmetric()

    def iotrace(self, i: int=0, o: int=0) -> Union[Tensor, Expression]:
        """
        :param i: Input index to be traced
        :param o: Output index to be trace
        :return: Trace of the tensor, when tracing a (1,1) tensor, this will just be a function, not a tensor object
        :raises IndexError: if one or both indices out of range

        Trace an i^th input and an o^th output of the Tensor
        """

        n_in = self.typ[0]
        n_out = self.typ[1]

        if i < -n_in or n_in <= i or o < -n_out or n_out <= o:
            raise IndexError('Index out of range')
        if n_in == 1 and n_out == 1:
            return sum(self.vals[k][k] for k in range(self.chart.dim))

        res = Tensor(self.chart, n_in - 1, n_out - 1)

        for pre in itertools.product(*[range(self.chart.dim) for i in range(i)]):
            for mid in itertools.product(*[range(self.chart.dim) for i in range(n_in + o - i - 1)]):
                for post in itertools.product(*[range(self.chart.dim) for i in range(n_out - o - 1)]):
                    for k in range(self.chart.dim):
                        res[pre + mid + post] += self[pre + (k,) + mid + (k,) + post]

        return res

    def alt(self) -> Form:
        """
        :return: Alternating version of this tensor as a Form
        :raises NotImplementedError: when the Tensor has outputs

        Only works if this Tensor is (k,0), compute the corresponding k-form alt(self)/k!
        """

        if self.typ[1] != 0:
            raise NotImplementedError("Can't compute the alternating form of a tensor with output")

        if self.size > self.chart.dim:
            return Form(self.chart, self.size)

        vals = {}
        for p in itertools.permutations(range(self.chart.dim), int(self.size)):
            ind = frozenset(p)
            vals[ind] = vals.setdefault(ind, 0) + sign(p) * self[p] / factorial(self.size)

        for ind in list(vals.keys()):
            try:
                vals[ind] = vals[ind].simplify_full()
            except:  # noqa
                pass
            try:
                if vals[ind] == 0: vals.pop(ind)
            except:  # noqa
                pass

        return Form(self.chart, self.size, vals)


class Form:
    """
    Class for differential forms on Manifold charts

    :Attributes:
        * **self.chart** -- The chart on which the form lives

        * **self.n** -- the order of the form

        * **self.vals** -- the values of the form, stored as a dictionary with frozenset entries

    .. math::
        &\\text{self} = f_I dx^I

        &dx^I = dx^{i_1} \\wedge dx^{i_2} \\wedge \\cdots \\wedge dx^{i_n}

        &i_1 < i_2 < \\cdots < i_n

        &\\text{self.vals}[\\text{frozenset}(I)] = f_I
    """

    def __init__(self, chart: Chart, n: int, vals: dict=None):
        """
        :param chart: Manifold chart on which this form lives
        :param n: The order of the chart
        :param vals: Values, optional, should be given as {frozenset(I): f\\ :sub:`I`} if not provided, the form is zero, but is still specifically viewed as having order n

        Create a differential form object.
        """
        self.chart = chart
        self.n = n
        self.vals = vals if vals else {}

    def __mul__(self, other: Union[Form, Expression]) -> Form:
        """
        :param other: The form with which to take the wedge product
        :return: wedge product of self and other or other*self if other is a function (or similar)
        :raises TypeError: if the two forms live on different charts

        Compute the wedge product of two forms, or multiply the form by a function
        """
        if isinstance(other, Form):
            if self.chart != other.chart:
                raise TypeError("Can't wedge forms on different charts")
            if self.n + other.n > self.chart.dim:
                return Form(self.chart, self.n + other.n)

            vals = {}

            for ind1 in self.vals:
                for ind2 in other.vals:
                    if ind1.isdisjoint(ind2):
                        ind1_list = list(ind1)
                        ind1_list.sort()
                        ind2_list = list(ind2)
                        ind2_list.sort()

                        ind_list = ind1_list + ind2_list
                        sgn = 1
                        for c in itertools.combinations(ind_list, 2):
                            if c[1] < c[0]: sgn *= -1
                        ind = frozenset(ind_list)
                        vals[ind] = vals.setdefault(ind, 0) + sgn * self.vals[ind1] * other.vals[ind2]
            for ind in list(vals.keys()):
                try:
                    vals[ind] = vals[ind].simplify_full()
                except:  # noqa
                    pass
                try:
                    if vals[ind] == 0: vals.pop(ind)
                except:  # noqa
                    pass
            return Form(self.chart, self.n + other.n, vals)
        else:
            return Form(self.chart, self.n, {I: other * self.vals[I] for I in self.vals})

    def __add__(self, other: Form) -> Form:
        """
        :param other: second form
        :return: self + other
        :raises TypeError: when impossible (other is not a form, a form of a different type,
        or a form on a different chart)

        Add two forms
        """
        if not isinstance(other, Form):
            raise TypeError("Can't add a non-form to a form")
        if self.chart != other.chart:
            raise TypeError("Can't add forms on different charts")
        if self.n != other.n:
            raise TypeError("Can't add forms of different orders")

        vals = self.vals.copy()
        for I in other.vals:
            vals[I] = vals.setdefault(I, 0) + other.vals[I]
        for I in list(vals.keys()):
            try:
                vals[I] = vals[I].simplify_full()
            except:  # noqa
                pass
            try:
                if vals[I] == 0: vals.pop(I)
            except:  # noqa
                pass

        return Form(self.chart, self.n, vals)

    def __sub__(self, other: Form) -> Form:
        """
        :param other: second form
        :return: self - other
        :raises TypeError: when impossible (other is not a form, a form of a different type,
        or a form on a different chart)

        Subtract two forms
        """
        if not isinstance(other, Form):
            raise TypeError("Can't subtract a non-form from a form")
        if self.chart != other.chart:
            raise TypeError("Can't subtract forms on different charts")
        if self.n != other.n:
            raise TypeError("Can't subtract forms of different orders")

        vals = self.vals.copy()
        for I in other.vals:
            vals[I] = vals.setdefault(I, 0) - other.vals[I]
        for I in list(vals.keys()):
            try:
                vals[I] = vals[I].simplify_full()
            except:  # noqa
                pass
            try:
                if vals[I] == 0: vals.pop(I)
            except:  # noqa
                pass

        return Form(self.chart, self.n, vals)

    def diff(self) -> Form:
        """
        :return: d(self)

        Compute the exterior derivative of a form
        """
        if self.n >= self.chart.dim:
            return Form(self.chart, self.n + 1)

        vals = {}

        for ind in self.vals:
            for i in range(self.chart.dim):
                if i not in ind:
                    sgn = 1
                    for j in ind:
                        if j < i: sgn *= -1
                    new_ind = set(ind.copy())
                    new_ind.add(i)
                    new_ind = frozenset(new_ind)
                    vals[new_ind] = vals.setdefault(new_ind, 0) + sgn * diff(self.vals[ind], self.chart.coords[i])

        for ind in list(vals.keys()):
            try:
                vals[ind] = vals[ind].simplify_full()
            except:  # noqa
                pass
            try:
                if vals[ind] == 0: vals.pop(ind)
            except:  # noqa
                pass

        return Form(self.chart, self.n + 1, vals)

    def cc(self, i: int) -> Form:
        """
        :param i: index of the coordinate vector field used to contract
        :return: contraction of self with the i\\ :sup:`th` coordinate vector
        :raises NotImplementedError: when done on a 0-form
        :raises IndexError: when `i` is out of range

        Contract the form with a coordinate vector field
        """
        if self.n < 1:
            raise NotImplementedError(f"Can't contract a {self.n}-form")
        if i < 0 or i >= self.chart.dim:
            raise IndexError('Index out of range')

        vals = {}
        for ind in self.vals:
            if i in ind:
                new_ind = set(ind)
                new_ind.remove(i)
                new_ind = frozenset(new_ind)
                sgn = 1
                for j in new_ind:
                    if j < i: sgn *= -1
                vals[new_ind] = vals.setdefault(new_ind, 0) + sgn * self.vals[ind]
        for ind in list(vals.keys()):
            try:
                vals[ind] = vals[ind].simplify_full()
            except:  # noqa
                pass
            try:
                if vals[ind] == 0: vals.pop(ind)
            except:  # noqa
                pass

        return Form(self.chart, self.n - 1, vals)

    def contract(self, vf: list) -> Form:
        """
        :param vf: the vector field to be used
        :return: contraction of self with the given vector field
        :raises NotImplementedError: when done on a 0-form
        :raises IndexError: if `vf` has wrong length

        Contract a form with a vector field, returns

        .. math:: \\imath_{\\text{vf}} (\\text{self})
        """
        if self.n < 1:
            raise NotImplementedError(f"Can't contract a {self.n}-form")
        if len(vf) != self.chart.dim:
            raise IndexError('Vector field has wrong size')

        res = Form(self.chart, self.n - 1)

        for i in range(self.chart.dim):
            if vf[i]:
                res += self.cc(i) * vf[i]

        return res

    def __str__(self) -> str:
        """
        :return: String representation of the form

        Format a string representation of the form
        """
        if not self.vals:
            return '0'
        s = ''
        for ind in self.vals:
            ind_list = list(ind)
            ind_list.sort()
            s += '(' + str(self.vals[ind]) + ') '
            for i in ind_list:
                s += 'd' + str(self.chart.coords[i]) + '^'
            s = s[:-1] + ' + '

        return s[:-3]

    def inner_prod(self, other: Form) -> Expression:
        """
        :param other: the second form
        :return: g(self,other)
        :raises TypeError: when impossible (other is not a form, other lives on a different chart,
            other is a form of a different type, the chart is not equipped with a metric)

        Compute the inner product of two forms
        """

        if not isinstance(other, Form):
            raise TypeError("Can't compute the inner product of a form and a non-form")
        if other.chart != self.chart:
            raise TypeError("Can't compute the inner product of forms on different charts")
        if other.n != self.n:
            raise TypeError("Can't compute the inner product of forms of differing indices")
        if not isinstance(self.chart, RiemChart):
            raise TypeError("Can't compute inner product without a metric")

        ip = 0
        for ind1 in self.vals:
            ind1_list = list(ind1)
            ind1_list.sort()
            for ind2 in other.vals:
                ind2_list = list(ind2)
                ind2_list.sort()
                m = Matrix([[self.chart.ginv[i][j] for i in ind1_list] for j in ind2_list])
                ip += self.vals[ind1] * other.vals[ind2] * det(m)
        return ip

    def hodge_star(self, vol: Expression=None) -> Form:
        """
        :param vol: The volume form coefficient , optional if not provided, uses the default Riemannian volume form
        :type vol: Expression
        :return: The hodge star of self
        :raises TypeError: When the chart is not equipped with a metric or the form has size larger
            than the dimension of the chart

        Compute the hodge star of a form. If `vol` is provided, the volume form is taken to be

        .. math:: \\text{vol}\\ dx^1\\wedge \\cdots \\wedge dx^n

        otherwise, uses the default Riemannian volume form:

        .. math:: \\text{vol} = \\sqrt{|\\text{det}(g)|}
        """

        if not isinstance(self.chart, RiemChart):
            raise TypeError("Can't compute the Hodge star without a metric")
        if self.n > self.chart.dim:
            raise TypeError("The form has size larger than the dimension of the chart")
        if not vol:
            vol = sqrt(abs(det(Matrix(self.chart.g))))

        vals = {}
        total = {i for i in range(self.chart.dim)}

        for ind1 in self.vals:
            ind1_list = list(ind1)
            ind1_list.sort()
            for c in itertools.permutations(total, int(self.n)):
                ind2 = frozenset(total - set(c))
                ind2_list = list(ind2)
                ind2_list.sort()
                eps = sign(list(c) + ind2_list)
                vals[ind2] = vals.setdefault(ind2, 0) + eps * vol * prod(
                    self.chart.ginv[c[i]][ind1_list[i]] for i in range(self.n)) * \
                             self.vals[ind1]

        for ind1 in list(vals.keys()):
            try:
                vals[ind1] = vals[ind1].simplify_full()
            except:  # noqa
                pass
            try:
                if vals[ind1] == 0: vals.pop(ind1)
            except:  # noqa
                pass
        return Form(self.chart, self.chart.dim - self.n, vals)

    def to_tensor(self) -> Tensor:
        """
        :return: The tensor version of this form

        Convert a differential form into a tensor
        """

        res = Tensor(self.chart, self.n, 0)

        for I in self.vals:
            for p in itertools.permutations(I):
                res[p] = sign(p) * self.vals[I]

        return res

    def codiff(self) -> Form:
        """
        :return: d\\ :sup:`*`\\ (self)
        :raises TypeError: when run on a 0-form or a form with order larger than the dimension of the chart.

        Compute the codifferential of this form
        """

        if self.n == 0:
            raise TypeError("Can't compute the codifferential of a 0-form")
        if self.n > self.chart.dim:
            raise TypeError("Form order is larger than the diimension of the chart")
        if self.n > 1:
            return self.to_tensor().diff().raise_ind(0).iotrace().alt() * (-1)
        elif self.n == 1:
            return Form(self.chart, 0, {frozenset({}): -self.to_tensor().diff().raise_ind(0).iotrace()})

    def hodge_laplace(self) -> Form:
        """
        :return: Hodge Laplacian of the form
        :raises TypeError: when the form has size larger than the dimension of the chart

        Compute the Hodge Laplacian

        .. math:: \\Delta^H(\\text{self}) = (d d^* + d^* d)(\\text{self})
        """

        if self.n > self.chart.dim:
            raise TypeError("Can't do Hodge Laplacian of a form of size larger than the dimension of the chart")
        elif self.n == self.chart.dim:
            return self.diff().codiff()
        elif self.n == 0:
            return self.codiff().diff()
        else:
            return self.codiff().diff() + self.diff().codiff()

def df(f: Expression, chart: Chart) -> Form:
    """
    :param f: function
    :type f: Expression
    :param chart: chart on which the function lives
    :type chart: Chart
    :return: df (as a form)

    Compute the differential of a function on a chart as a form
    """

    return Form(chart, 1, {frozenset({k}): diff(f, chart.coords[k]) for k in range(chart.dim)})


def laplace(f: Union[Expression, Tensor, Form], chart: Chart=None) -> Union[Expression, Tensor, Form]:
    """
    :param f: a function, tensor, or form
    :param chart: chart, optional if f in a tensor or form, required if f is a function
    :return: the Laplacian of f
    :raises NotImplementedError: when trying to compute the Laplacian of a function without specifying a chart.

    Compute the laplace of f, a function, tensor, or form
    """

    if isinstance(f, Tensor):
        n_in, n_out = f.typ
        return f.diff().diff().raise_ind(0).iotrace(0, n_out)
    if isinstance(f, Form):
        if f.n > 0:
            return laplace(f.to_tensor()).alt()
        else:
            return laplace(f.vals[frozenset({})], f.chart)
    elif not chart:
        raise NotImplementedError("Can't compute laplace of a function without a chart")
    else:
        return df(f, chart).to_tensor().diff().iotrace()


def hessian(f: Expression, chart: Chart) -> Tensor:
    """
    :param f: function
    :param chart: chart
    :return: Hess(f) as a (2,0)-tensor

    Compute the hessian of a function f on a given chart

    .. math:: \\text{Hess}(f) = \\nabla\\nabla f
    """

    return df(f, chart).to_tensor().diff()
