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
    r, t = var('r t')
    # Create M as a 2-dimensional coordinate chart with coordinates [r, t] and the aforementioned metric
    M = RiemChart(2, [r, t], [[1, 0], [0, sin(r)^2]])
    # Check sectional curvature
    M.sec([1,0],[0,1]) # Output: 1
    # Check the Ricci tensor values
    M.ric_t.vals # Output: [[1, 0], [0, sin(r)^2]]

In the above we created a 2-dimensional Riemannian chart with coordinates `r` and `t` and the metric

.. math:: g=dr^2 + \sin^2 r \ dt^2

Equivalently given by the matrix

.. math:: \begin{pmatrix}1 & 0\\0 & \sin^2 r\end{pmatrix}

This is the round sphere metric in polar coordinates, `r` measures distance from one pole, and `t` is along the lines
of latitude.

We then computed sectional curvature of the plane spanned by the vectors (1,0) and (0,1), the output is 1 as expected.

We then computed the full Ricci tensor, getting it as a matrix

.. math:: \begin{pmatrix}1 & 0\\0 & \sin^2 r\end{pmatrix}

which is the same as the starting metric.
