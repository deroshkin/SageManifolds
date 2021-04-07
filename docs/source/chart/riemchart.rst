======================================================================================
``RiemChart`` -- Coordinate chart with a (pseudo-)Riemannian metric
======================================================================================

.. toctree::
    :maxdepth: 1

.. autoclass:: manifolds.riemannian.RiemChart
    :members: __init__, inner, connection, curv, sec, ricci, path_derivative, parallel_transport

=============
Examples
=============

:Example 1:

.. math:: \mathbb{S}^2

.. code-block:: python

    # Initialize variables r and t
    r, s = var('r s')
    # Create M as a 2-dimensional coordinate chart with coordinates [r, s] and the aforementioned metric
    M = RiemChart(2, [r, s], [[1, 0], [0, sin(r)^2]])
    # Check sectional curvature
    M.sec([1,0],[0,1]) # Output: 1
    # Check the Ricci tensor values
    M.ric_t.vals
    # Output: [[1, 0], [0, sin(r)^2]]
    # Check the connection
    M.connection([1,0],[1,0])
    # Output: [0,0] shows that [t,0] is a geodesic
    # Check the parallel transport along the path r=$\pi/4$, s=t
    M.parallel_transport([pi/4, t])
    # Output: ([f1(t), f2(t)], [-1/2*f2(t) + diff(f1(t), t) == 0, f1(t) + diff(f2(t), t) == 0])

In the above we created a 2-dimensional Riemannian chart with coordinates `r` and `s` and the metric

.. math:: g = dr^2 + \sin^2 r \ ds^2

Equivalently given by the matrix

.. math:: \begin{pmatrix}1 & 0\\0 & \sin^2 r\end{pmatrix}

This is the round sphere metric in polar coordinates, `r` measures distance from one pole, and `s` is along the lines
of latitude.

We then computed sectional curvature of the plane spanned by the vectors (1,0) and (0,1), the output is 1 as expected.

We then computed the full Ricci tensor, getting it as a matrix

.. math:: \begin{pmatrix}1 & 0\\0 & \sin^2 r\end{pmatrix}

which is the same as the starting metric.

Next we check that

.. math:: \nabla_{\frac{d}{dr}} \frac{d}{dr} = 0.

The last thing we checked is parallel transport along the curve :math:`r=\pi/4,\ s=t`. The output tells us that
:math:`f_1(t)\frac{d}{dr} + f_2(t)\frac{d}{ds}` is parallel when :math:`f_1,f_2` satisfy

.. math::  f_1'(t) - \frac{f_2(t)}{2} &= 0\\
           f_2'(t) + f_1(t) &= 0.
