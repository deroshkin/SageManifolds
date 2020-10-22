======================================
``Chart`` -- A basic coordinate chart
======================================

.. autoclass:: manifolds.Base.Chart
    :members: __init__, bracket

-----

:Example:

.. code-block:: python

    # Initialize variables x, y, and z
    x, y, z = var('x y z')
    # Create M as a 3-dimensional coordinate chart with coordinates [x, y, z]
    M = Chart(3, [x, y, z])
    #Compute the Lie bracket of the vector fields (x^2, z, 0) and (y, x, z)
    M.bracket([x^2, z, 0], [y, x, z]) # Output: [-2*x*y + z, x^2 - z, 0]

In the above we created a 3-dimensional chart with coordinates `x`, `y`, `z` and then asked the computer to calculate

.. math:: \left[x^2\frac{d}{dx} + z\frac{d}{dy},\ y\frac{d}{dx} + x\frac{d}{dy} + z\frac{d}{dz}\right]

and got

.. math:: (-2xy+z)\frac{d}{dx} + (x^2-z)\frac{d}{dy}