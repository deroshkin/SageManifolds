# Manifolds

This python package is designed to work with [sage](https://www.sagemath.org/). Attempting to run it in plain python
will not work!

All the libraries in this package are designed to work in local coordinates within a single coordinate chart, with the
focus on explicit computations. The three base object for all of this are: `Chart`, `Tensor`, and `Form`.

## `Chart` Objects

All computations which use this package require a base `Chart` object. In additions to the base `Chart` class, there
are several subclasses that are more convenient to use:

* `RiemChart` - A subclass of `Chart` for charts on Riemannian manifolds.
* `ComplexChart` - A subclass of `Chart` for charts with complex coordinates, which are split into holomorphic and
  antiholomorphic.
* `KahlerChart` - A subclass of both `ComplexChart` and `RiemChart` classes to model Kähler manifolds, built from a
  Kähler potential function.
* `GenChart` - A meta-class which is a subclass of `RiemChart`, meant for manifolds with a metric and a torsion-free
  connection, which need not be compatible with the metric. There are 2 implementations of this provided in the
  package: `DensChart` and `StatChart`.
  
Note that for many computations either `RiemChart` or a subclass of it is required.

## `Tensor` Objects

The `Tensor` class is used to model arbitrary tensors on any chart, one of the few computations that can be performed
in general is tracing the tensor's upper and lower indices. If the tensor lives on a `RiemChart`, one can also
raise/lower indices and compute the covariant derivative.

## `Form` Objects

The `Form` class is used to model arbitrary forms on any chart. Each form is stored using a dictionary in the `.vals`
property, which is structured as `{frozenset({i1,i2,...,ik}): f_{i1,i2,...,ik}}` to represent `f_{i1,i2,...,ik}
dx^{i1} \wedge dx^{i2} \wedge ... \wedge dx^{ik}`.
