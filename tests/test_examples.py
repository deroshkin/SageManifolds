import os
import sys
import unittest
from random import random
from sage.all import Matrix, var, cos, sin, sqrt, arctan, pi

sys.path.insert(0, os.path.abspath('..'))
from manifolds.examples import hyperbolic_half_space, sphere_conformal, sphere_polar
from manifolds.examples import complex_projective_space


def rnd(a=None, b=None):
    """generate a random number in the interval [a,b],
    a = None is treated as -infinity
    b = None is treated as +infinity
    for finite ranges uses a uniform distribution, for infinite, uses arctan of a uniform ditribution"""

    if a is None and b is None:
        return arctan((random() - 0.5) * pi).n()
    elif a is None:
        return b - arctan(random() * pi / 2).n()
    elif b is None:
        return a + arctan(random() * pi / 2).n()
    else:
        return a + random() * (b - a)


class H3Tests(unittest.TestCase):
    def setUp(self):
        self.H3 = hyperbolic_half_space(3)

    def testGinv(self):
        """Testing that M.g and M.ginv are inverses"""
        self.assertEqual(self.H3.g * self.H3.ginv, Matrix.identity(3))

    def testVars(self):
        """Testing variable names in hyperbolic space"""
        self.assertEqual(tuple(self.H3.coords), var('x1 x2 x3'))

    def testSec(self):
        """Testing principal sectional curvatures in hyperbolic space"""
        self.assertEqual(self.H3.sec([1, 0, 0], [0, 1, 0]).simplify_full(), -1)
        self.assertEqual(self.H3.sec([1, 0, 0], [0, 0, 1]).simplify_full(), -1)
        self.assertEqual(self.H3.sec([0, 1, 0], [0, 0, 1]).simplify_full(), -1)

    def testEinstein(self):
        """Testing that the hyperbolic space is Einstein"""
        self.assertEqual(Matrix(self.H3.ric_t.vals).simplify_full(), -2 * self.H3.g)

    def testGeodesics(self):
        """Testing geodesics of the hyperbolic space"""
        t = var('t')
        self.assertEqual(self.H3.path_derivative([0, 0, t], [0, 0, t]), [0, 0, 0])
        self.assertEqual(self.H3.path_derivative([cos(t), 0, sin(t)], [-sin(t) ** 2, 0, cos(t) * sin(t)]), [0, 0, 0])
        self.assertEqual(self.H3.path_derivative([2 * cos(t), cos(t), sqrt(5) * sin(t)],
                                                 [-2 * sqrt(5) * sin(t) ** 2, -sqrt(5) * sin(t) ** 2,
                                                  5 * cos(t) * sin(t)]), [0, 0, 0])


class S4ConfTests(unittest.TestCase):
    def setUp(self):
        self.S4 = sphere_conformal(4)

    def testVars(self):
        """Testing variable names of conformal S4"""
        self.assertEqual(tuple(self.S4.coords), var('x1 x2 x3 x4'))

    def testGinv(self):
        """Testing that M.g and M.ginv are inverses"""
        self.assertEqual(self.S4.g * self.S4.ginv, Matrix.identity(4))

    def testSec(self):
        """Testing principal sectional curvatures"""
        self.assertEqual(self.S4.sec([1, 0, 0, 0], [0, 1, 0, 0]).simplify_full(), 1)
        self.assertEqual(self.S4.sec([1, 0, 0, 0], [0, 0, 1, 0]).simplify_full(), 1)
        self.assertEqual(self.S4.sec([1, 0, 0, 0], [0, 0, 0, 1]).simplify_full(), 1)
        self.assertEqual(self.S4.sec([0, 1, 0, 0], [0, 0, 1, 0]).simplify_full(), 1)
        self.assertEqual(self.S4.sec([0, 1, 0, 0], [0, 0, 0, 1]).simplify_full(), 1)
        self.assertEqual(self.S4.sec([0, 0, 1, 0], [0, 0, 0, 1]).simplify_full(), 1)

    def testEinstein(self):
        """Testing that the sphere is Einstein"""
        self.assertEqual(Matrix(self.S4.ric_t.vals).simplify_full(), 3 * self.S4.g)


class S4PolarTests(unittest.TestCase):
    def setUp(self):
        self.S4 = sphere_polar(4)

    def testVars(self):
        """Testing variable names of conformal S4"""
        self.assertEqual(tuple(self.S4.coords), var('r t1 t2 t3'))

    def testGinv(self):
        """Testing that M.g and M.ginv are inverses"""
        self.assertEqual(self.S4.g * self.S4.ginv, Matrix.identity(4))

    def testSec(self):
        """Testing principal sectional curvatures"""
        self.assertEqual(self.S4.sec([1, 0, 0, 0], [0, 1, 0, 0]).simplify_full(), 1)
        self.assertEqual(self.S4.sec([1, 0, 0, 0], [0, 0, 1, 0]).simplify_full(), 1)
        self.assertEqual(self.S4.sec([1, 0, 0, 0], [0, 0, 0, 1]).simplify_full(), 1)
        self.assertEqual(self.S4.sec([0, 1, 0, 0], [0, 0, 1, 0]).simplify_full(), 1)
        self.assertEqual(self.S4.sec([0, 1, 0, 0], [0, 0, 0, 1]).simplify_full(), 1)
        self.assertEqual(self.S4.sec([0, 0, 1, 0], [0, 0, 0, 1]).simplify_full(), 1)

    def testEinstein(self):
        """Testing that the sphere is Einstein"""
        self.assertEqual(Matrix(self.S4.ric_t.vals).simplify_full(), 3 * self.S4.g)


class CP2Tests(unittest.TestCase):
    def setUp(self):
        self.CP2 = complex_projective_space(2)

    def testVars(self):
        """Testing variable names of CP2"""
        self.assertEqual(tuple(self.CP2.coords), var('z1 z2 z1b z2b'))

    def testGinv(self):
        """Testing that M.g and M.ginv are inverses"""
        self.assertEqual(self.CP2.g * self.CP2.ginv, Matrix.identity(4))

    def testzzb(self):
        """Test that the metric has g(d/dz, d/dz) = g(d/dzb, d/dzb) = 0"""
        for i in range(2):
            for j in range(2):
                self.assertEqual(self.CP2.g[i][j], 0)
                self.assertEqual(self.CP2.g[2 + i][2 + j], 0)

    def testEinstein(self):
        """Testing that the complex projective plane is Einstein"""
        self.assertEqual(Matrix(self.CP2.ric_t.vals).simplify_full(), 6 * self.CP2.g)


class CP2RealTests(unittest.TestCase):
    def setUp(self):
        self.CP2R = complex_projective_space(2).to_riem_chart()

    def testVars(self):
        """Test variable names"""
        self.assertEqual(tuple(self.CP2R.coords), var('x1 y1 x2 y2'))

    def testGinv(self):
        """Testing that M.g and M.ginv are inverses"""
        self.assertEqual(self.CP2R.g * self.CP2R.ginv, Matrix.identity(4))

    def testEinstein(self):
        """Testing that the complex projective plane is Einstein"""
        self.assertEqual(Matrix(self.CP2R.ric_t.vals).simplify_full(), 6 * self.CP2R.g)

    def testQuarterPinch(self):
        """Test five random tangent 2-planes to verify that sectional curvature is in [1,4]"""

        var('x1 y1 x2 y2')

        for _ in range(5):
            pt = [rnd() for _ in range(4)]
            v1 = [rnd(-1, 1) for _ in range(4)]
            v2 = [rnd(-1, 1) for _ in range(4)]
            sec = self.CP2R.sec(v1, v2)(x1=pt[0], y1=pt[1], x2=pt[2], y2=pt[3])
            self.assertGreaterEqual(sec, 1)
            self.assertLessEqual(sec, 4)


if __name__ == '__main__':
    unittest.main()
