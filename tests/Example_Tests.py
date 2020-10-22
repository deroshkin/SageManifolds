import os, sys

sys.path.insert(0, os.path.abspath('..'))
from manifolds.Examples import hyperbolic_half_space
import unittest
from sage.all import Matrix, var, cos, sin, sqrt


class H3Tests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.H3 = hyperbolic_half_space(3)

    def testSec(self):
        self.assertEqual(self.H3.sec([1, 0, 0], [0, 1, 0]).simplify_full(), -1)
        self.assertEqual(self.H3.sec([1, 0, 0], [0, 0, 1]).simplify_full(), -1)
        self.assertEqual(self.H3.sec([0, 1, 0], [0, 0, 1]).simplify_full(), -1)

    def testEinstein(self):
        self.assertEqual(Matrix(self.H3.ric_t.vals).simplify_full(), -2 * self.H3.g)

    def testGeodesics(self):
        t = var('t')
        self.assertEqual(self.H3.path_derivative([0, 0, t], [0, 0, t]), [0, 0, 0])
        self.assertEqual(self.H3.path_derivative([cos(t), 0, sin(t)], [-sin(t) ** 2, 0, cos(t) * sin(t)]), [0, 0, 0])
        self.assertEqual(self.H3.path_derivative([2 * cos(t), cos(t), sqrt(5) * sin(t)],
                                                 [-2 * sqrt(5) * sin(t) ** 2, -sqrt(5) * sin(t) ** 2,
                                                  5 * cos(t) * sin(t)]), [0, 0, 0])


# if __name__ == '__main__':
#     unittest.main()
