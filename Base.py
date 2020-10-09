from sage.all import var, diff


class Chart:
    """
    The base class for all Chart types, used for handling a single coordinate chart on a manifold.

    :Attributes:
        * **self.n** -- The dimension of the manifold chart

        * **self.coords** -- The variables being used as the coordinates on this chart

    """

    def __init__(self, dim: int, coords: list=None):
        """
        :param dim: real dimension of the manifold - becomes self.n
        :param coords: names of the coordinate variables, optional, defaults to [`x1`, `x2`, `x3`, ..., `xn`]
        :raises IndexError: if the length of the provided list of coordinate variables does not match the dimension

        Create a manifold chart object
        """
        self.dim = dim
        if coords:
            if len(coords) != dim:
                raise IndexError('The number of coordinate variables must equal the dimension')
            self.coords = coords
        else:
            self.coords = [var('x' + str(i + 1)) for i in range(self.dim)]

    def bracket(self, vf1: list, vf2: list) -> list:
        """
        :param vf1: First vector field
        :param vf2: Second vector field
        :return: The bracket [`vf1` , `vf2`]
        :raises IndexError: if one (or both) of the vector fields has the wrong size

        Compute the Lie bracket of two vector fields
        """

        if len(vf1) != self.dim or len(vf2) != self.dim:
            raise IndexError('Vector fields must have length = dimension')
        vf_res = []
        for j in range(self.dim):
            c = 0
            for i in range(self.dim):
                c += vf1[i] * diff(vf2[j], self.coords[i]) - vf2[i] * diff(vf1[j], self.coords[i])
            vf_res.append(c)
        return vf_res

## Local Variables:
## mode: Python
## End:
