LinAlg-magma
============

LinAlg-hmatrix provides a GPU-powered backend for the Haskell library
[LinAlg](https://github.com/bmsherman/LinAlg/). The backend has the
following dependencies:

1. [cublas](https://github.com/bmsherman/cublas), which provides FFI
bindings for the CUBLAS library.

2. [magma-gpu](https://github.com/bmsherman/magma-gpu), which provides FFI
bindings for the MAGMA GPU library.

3. [LinAlg](https://github.com/bmsherman/LinAlg/), which provides the
purely functional interface for computations.

Usage
-----

To use the purely functional LinAlg interface, simply 
import `Numeric.LinAlg.Magma`.

This library also provides a lower-level API in `Numeric.LinAlg.Internal`
and `Numeric.LinAlg.Mutable` that allows for mixing and
matching of stateful and purely functional computations on mutable and
immutable vector and matrix types. The mutable interface uses an `ST`-like
monadic interface. The type system allows for mutable operations to read
from either immutable or mutable datatypes, so you can easily mix the two
and get the best of both worlds.

I will hopefully describe and document this system in more detail in the
future!
