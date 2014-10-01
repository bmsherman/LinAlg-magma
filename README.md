LinAlg-magma
============

LinAlg-magma provides a GPU-powered backend for the Haskell library
[LinAlg](https://github.com/bmsherman/LinAlg/). Therefore, it allows
execution of purely functional numerical linear algebra operations
on the GPU. Matrix and vector sizes are checked statically.

Additionally, a mutable interface is provided (at a higher level of 
abstraction than the FFI bindings) which can be used in conjunction with
the immutable one in order to achieve better performance (in particular,
to avoid unnecessary copying).

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

### Documentation

[The haddock documentation](http://bmsherman.github.io/haddock/LinAlg-magma/)
is relatively complete. In particular, there is a [semi-thorough
explanation of of the mutable and immutable interface work together](http://bmsherman.github.io/haddock/LinAlg-magma/Numeric-LinAlg-Magma-Mutable.html#g:1).

### Installation

LinAlg-magma has three dependencies which are not yet available on Hackage.
Before installing LinAlg-magma, you must install these three packages:

1. [cublas](https://github.com/bmsherman/cublas), which provides FFI
bindings for the CUBLAS library. This package is now available on
Hackage, and should be automatically installed.

2. [magma-gpu](https://github.com/bmsherman/magma-gpu), which provides FFI
bindings for the MAGMA GPU library. This package is not on Hackage, and
must be manually installed.

3. [LinAlg](https://github.com/bmsherman/LinAlg/), which provides the
purely functional interface for computations. This package is not on 
Hackage, and must be manually installed.
