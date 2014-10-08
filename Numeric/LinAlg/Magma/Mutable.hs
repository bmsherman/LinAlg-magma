{-# LANGUAGE FlexibleInstances, RankNTypes, GeneralizedNewtypeDeriving #-}
{-# LANGUAGE DataKinds, KindSignatures, TypeOperators #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE FlexibleContexts #-}

{- | This module provides a mutable interface for performing GPU
computations using the CUBLAS and MAGMA libraries. It provides a higher
level interface to CUBLAS and MAGMA than is provided by the corresponding
FFIs, but a lower-level interface than LinAlg. -}

module Numeric.LinAlg.Magma.Mutable (

  -- * The read-write monad
  RW,
  makeRWD, makeRWV, makeRWM, runRW, creatingV, creatingM,
  -- * Data types
  CNum (..),
  -- ** Mutable
  VecP (..), MatP (..), DVecP (..), DMatP (..),
  -- ** Immutable
  Ice, Mat, Vec,
  -- * Data transfer
  fromVectP, fromVectsP,
  toVectP, toListsP, toVectsP,
  makeVecP, makeMatP,
  copy, makeCopyVecP, makeCopyMatP,
  memtrans, fillWithColumns,
  -- * BLAS-like functions
  -- | See documentation for CUBLAS and MAGMA for more detailed
  -- documentation.

  -- ** CUBLAS
  -- | See <http://docs.nvidia.com/cuda/cublas/ CUBLAS documentation>.
  dot, scal, axpy,
  gemv, gemm, 

  trsv, trsm,
  geam,
  dgmm,
  ger, syrk,
  -- ** MAGMA
  -- | See <http://icl.cs.utk.edu/magma/docs/ MAGMA documentation>.
  gesv, potrf,

  -- * Miscellaneous
  asColumns, asVecP, dimCheck,
  handle, magmaHandle,
  setZeroMatP, takeDiagP,

) where

import Control.Applicative (Applicative)

import Data.List.Split (chunksOf)
import Data.Complex (Complex (..))
import Data.Proxy (Proxy (..))

import Foreign.CUDA

import Foreign.C.Types (CFloat, CDouble)
import Foreign.Ptr (Ptr)
import Foreign.Storable (Storable)
import Foreign.Marshal.Array (withArray)

import Foreign.CUDA.Cublas.Types
import qualified Foreign.CUDA.Cublas.FFI as C
import qualified Foreign.CUDA.Cublas as C
import qualified Foreign.CUDA.Magma as Magma
import qualified Foreign.CUDA.Magma.Types as Magma

import GHC.TypeLits 

import Numeric.LinAlg.SNat (SNat, snat, lit, times, minus)
import Numeric.LinAlg.Vect (Vect (MkVect))
import qualified Numeric.LinAlg.Vect as V

import System.Mem.Weak (addFinalizer)
import System.IO.Unsafe (unsafePerformIO)

-- | This class represents the elements on which we can operate.
class (C.Cublas e, Magma.Magma e, Storable e, Floating e) => CNum e where
instance CNum CFloat
instance CNum CDouble
instance CNum (Complex CFloat)
instance CNum (Complex CDouble)

-- | A monad that that behaves much like the 'ST' monad. The first
-- type argument is a phantom type that represents the state.
-- A value of type @ RW s a @ is a computation which performs mutable
-- operations on the GPU and returns a value of type 'a'.
--
-- Like in the 'ST' monad, the first type argument represents the state
-- thread of the computation. Computations which produce values
-- whose type is independent of this state thread are pure, and may be
-- extracted from the monad (see the Internal module). /Unlike/ 'ST',
-- however, we may /still/ extract values whose types do depend on the
-- state thread (i.e., mutable values). However, when we extract these
-- values into pure Haskell-space, we set the state-thread to the void
-- type 'Ice', which represents a /frozen/ state thread, where the value
-- may no longer be modified. This is enforced by the type-system.
--
-- This allows us to coherently treat mutable and immutable versions of
-- the same datatype! We may have mutable computations which may /read/
-- from either mutable or immutable value, but only write to mutable
-- values. In our types, we declare that we may read from a value which is
-- in any state thread; in reality there are only two possible state threads
-- in scope - that of /this/ mutable computation, and the frozen state
-- thread, 'Ice'. However, when we write to a value, that value must be
-- in the same state thread as our computation (and cannot be immutable!).
--
-- For example, consider the 'copy' operation, which reads from the first
-- vector and writes to the second. The 'copy' operation produces a
-- computation in the state thread @s@, and so the vector which is written
-- to must also be in the state thread @s@. The vector which is read from
-- may /either/ be in the state thread @s@ (meaning it is mutable as well)
-- or have state thread 'Ice' (indicating that it is now immutable).
-- Therefore, its type is @t@, which is implicitly universally quantified,
-- and so may be instantiated by either @s@ or 'Ice'. 
newtype RW s a = RW { unRW :: IO a } deriving (Functor, Applicative, Monad)

-- | A type for vectors which reside on the GPU.
-- The first type argument is a phantom type that is used similarly to the
-- state type in the 'ST' monad.
data VecP s (n :: Nat) a = VecP
  { ptrV   :: DevicePtr a -- ^ a pointer to the payload on the GPU
  , len    :: SNat n -- ^ the number of elements in the vector
  , stride :: Int -- ^ the offset in the payload from one element of the 
                  -- vector to the next
  }

data DVecP s a where
  DVecP :: VecP s n a -> DVecP s a

data DMatP s a where
  DMatP :: MatP s m n a -> DMatP s a

-- | A type for matrices which reside on the GPU. Matrices are stored
-- in column-major format, which is the format preferred by CUBLAS.
-- The first type argument is a phantom type that is used similarly to the
-- state type in the 'ST' monad.
data MatP s (m :: Nat) (n :: Nat) a = MatP
  { ptrM :: DevicePtr a -- ^ a pointer to the payload on the GPU
  , dim  :: (SNat m, SNat n) -- ^ the dimensions of the matrix (rows, columns)
  , ld   :: Int -- ^ the leading dimension of the matrix
  }


-- | A void type which is used in the "state thread" type variable in
-- mutable structures. As the name suggests, types which have 'Ice' as
-- their state thread are immutable. This immutability is enforced by the
-- type system. The only way in which mutable datatypes in the RW monad
-- are allowed to leave is by having their state thread type variables
-- coerced to 'Ice'.
data Ice

-- | An immutable version of the 'MatP' primitive matrix data type.
type Mat = MatP Ice

-- | An immutable version of the 'VecP' primitive vector data type.
type Vec = VecP Ice

-- | If a computation in the 'RW' monad is valid for /all/ state threads,
-- and produces a state-/dependent/ type, then we may extract out the
-- value as a pure computation, as long as we coerce the state thread
-- type variable to 'Ice', indicating that the value is now immutable.
-- We do /not/ need to make a copy of the mutable value; rather, it
-- may no longer be mutated.
makeRWD :: (forall s. RW s (dyn s e)) -> dyn Ice e
makeRWD (RW x) = unsafePerformIO x

makeRWV :: (forall s. RW s (VecP s n e)) -> Vec n e
makeRWV (RW x) = unsafePerformIO x

makeRWM :: (forall s. RW s (MatP s n m e)) -> Mat n m e
makeRWM (RW x) = unsafePerformIO x

-- | If a computation in the 'RW' monad is valid for /all/ state threads,
-- and produces a type which is independent of the state thread, then we
-- may extract out that value as a pure computation.
runRW :: (forall s. RW s a) -> a
runRW (RW x) = unsafePerformIO x

-- | A convenient function for creating immutable data with mutable
-- operations. The first argument is an operation to construct a data
-- structure, the second argument operates on the first once
-- it has been constructed, and the resulting data structure is returned
-- as an immutable result.
creatingV :: (forall s. RW s (VecP s n e)) -> (forall s. VecP s n e -> RW s b) 
         -> Vec n e
creatingV creator f = makeRWV $ do
  x <- creator
  f x
  return x

creatingM :: (forall s. RW s (MatP s n m e)) -> (forall s. MatP s n m e -> RW s b) 
         -> Mat n m e
creatingM creator f = makeRWM $ do
  x <- creator
  f x
  return x

-- | A global handle created when CUBLAS initializes that is passed
-- to all CUBLAS calls.
handle :: Handle
handle = unsafePerformIO $ do
  h <- C.create
  putStrLn "CUBLAS initialized."
  addFinalizer handle (C.destroy h >> putStrLn "CUBLAS finalized")
  return h

-- | A /fake/ handle for MAGMA, because we similarly must initialize
-- MAGMA prior to calling any MAGMA functions.
magmaHandle :: Maybe ()
magmaHandle = unsafePerformIO $ do
  Magma.initialize
  putStrLn "MAGMA initialized."
  addFinalizer handle (Magma.finalize >> putStrLn "MAGMA finalized")
  return $ Just ()


mkFinalized :: IO a -> (a -> IO ()) -> IO a
mkFinalized create destroy = do
  x <- create
  addFinalizer x (destroy x)
  return x

-- | Make a new (garbage-collected) array on the GPU with a given length.
-- Elements are uninitialized.
makeArray :: Storable a => Int -> IO (DevicePtr a)
makeArray n = mkFinalized (mallocArray n) free

-- | Make a copy of a GPU array, specifying the array's length manually.
makeCopyArray :: Storable a => Int -> DevicePtr a -> IO (DevicePtr a)
makeCopyArray n pa = do
  pb <- makeArray n
  copyArray n pa pb
  return pb

-- | Make a new matrix on the GPU with given dimensions. Elements are
-- uninitialized.
makeMatP :: Storable a
  => (SNat m, SNat n) -> RW s (MatP s m n a)
makeMatP dim@(m,n) = do
  p <- RW $ makeArray (m' * natInt n)
  return $ MatP p dim m'
  where m' = natInt m

-- | Make a copy of a matrix.
makeCopyMatP :: Storable a => MatP t m n a -> RW s (MatP s m n a)
makeCopyMatP (MatP pa dim@(m,n) lda) = 
  if m' == lda 
    then do
      mat@(MatP pb _ _) <- makeMatP dim
      RW $ copyArray (m' * natInt n) pa pb
      return mat
    else error "makeCopyMatP: Not yet implemented"
  where m' = natInt m

natInt :: SNat n -> Int
natInt = snat

-- | Make a copy of a vector.
copy :: C.Cublas a => VecP t n a -> VecP s n a -> RW s ()
copy (VecP px nx incx) (VecP py ny incy) = dimCheck "copy" [[nx,ny]] $
  RW $ C.copy handle (natInt nx) px incx py incy

-- | Create a new vector from a list.
fromVectP :: Storable a => Vect n a -> RW s (VecP s n a)
fromVectP xs = do
    px <- RW $ makeArray (natInt nx)
    RW $ pokeListArray (V.toList xs) px
    return (VecP px nx 1)
  where nx = V.length xs

-- | Create a new matrix from a list of lists. The matrix is loaded in
-- /column major/ format!
fromVectsP :: Storable a => Vect n (Vect m a) -> RW s (MatP s m n a)
fromVectsP vect = do
    pa <- RW $ makeArray (natInt (times m n))
    RW $ pokeListArray (concat (map V.toList (V.toList vect))) pa
    return $ MatP pa (m, n) (natInt m)
  where
  m = V.length (head (V.toList vect))
  n = V.length vect

-- | Read the elements of a vector into a list.
toVectP :: CNum e => VecP t n e -> RW s (Vect n e)
toVectP (VecP px nx 1) = fmap MkVect $ RW (peekListArray (natInt nx) px)
toVectP v = makeCopyVecP v >>= toVectP

-- | Read the elements of a matrix into a list of lists in column-major
-- format.
toListsP :: Storable a => MatP t m n a -> RW s [[a]]
toListsP c@(MatP pa (m,n) lda) = 
  if m' == lda 
    then RW . fmap (chunksOf m') $ peekListArray (m' * natInt n) pa
    else error "toListsP: Not yet implemented"
  where m' = natInt m


toVectsP :: Storable a => MatP t m n a -> RW s (Vect n (Vect m a))
toVectsP c = fmap (MkVect . map MkVect) (toListsP c)

-- | View the diagonal of a matrix as a vector. Note that this
-- does not make a copy (as is evident from the type)!
takeDiagP :: MatP s n n e -> VecP s n e
takeDiagP a@(MatP pa (ma,na) lda) = let n' = min ma na in 
  VecP pa n' (lda + 1)

-- | Make a new vector of a given length. Elements remain unitialized.
makeVecP :: Storable a => SNat n -> RW s (VecP s n a)
makeVecP n = do
  p <- RW $ makeArray (natInt n)
  return (VecP p n 1)

-- | Set all of the elements in a matrix to 0.
setZeroMatP :: CNum a => MatP s m n a -> RW s ()
setZeroMatP a = geam 0 (a, N) 0 (a, N) a

-- | Make a copy of a vector.
makeCopyVecP :: (Storable a, C.Cublas a) => VecP t n a -> RW s (VecP s n a)
makeCopyVecP (VecP px nx incx) = do
  py <- RW $ makeArray nx'
  RW $ C.copy handle nx' px incx py 1
  return $ VecP py nx 1
  where
  nx' = natInt nx

-- | Cholesky decomposition of a positive-definite matrix.
potrf :: CNum a => (MatP s n n a, FillMode) -> RW s ()
potrf (MatP pa (n,n') lda, fill) = dimCheck "potrf" [[n, n']] $ do
  (_,res) <- magmaHandle `seq` RW $ Magma.potrf (toMagma fill) (natInt n) pa lda
  return $ if res /= 0 then error "Mutable.potrf: MAGMA potrf failed" else ()
  where
  toMagma Upper = Magma.Upper
  toMagma Lower = Magma.Lower

-- | General matrix-matrix linear system solver.
-- Warning! 'gesv' acts destructively on /both/ matrices.
gesv :: CNum a => MatP s n n a -> MatP s n m a -> RW s ()
gesv (MatP pa (n,n') lda) (MatP pb (n'', m) ldb) = 
  dimCheck "gesv" [[n, n', n'']] $ do
    (_,res) <- magmaHandle `seq` RW $ 
      withArray (replicate (min mi ni) 0) $ \ipiv -> 
        Magma.gesv ni mi pa lda ipiv pb ldb
    return $ if res /= 0 
      then error "Mutable.gesv: MAGMA gesv failed" 
      else ()
  where
  mi = natInt m
  ni = natInt n

-- | Solution of a triangular linear system against a vector.
trsv :: CNum a => (MatP t n n a, FillMode, Operation, DiagType)
     -> VecP s n a -> RW s ()
trsv (MatP pa (n,n') lda, uplo, transa, diag) (VecP px nx incx) = 
  dimCheck "trsv" [[n, n', nx]] . RW $
    C.trsv handle uplo transa diag (natInt n) pa lda px incx

-- | Solution of a triangular linear system against a matrix.
-- XXX: make this type more specific
trsm :: CNum a => (MatP t n n a, SideMode, FillMode, Operation, DiagType)
     -> a -> MatP s m p a -> RW s ()
trsm (MatP pa (ma,na) lda, side, uplo, transa, diag) alpha
     (MatP px (m, n) ldb) = 
  dimCheck "trsm" [[natInt ma,natInt na,n']] . RW $
    C.trsm handle side uplo transa diag (natInt m) (natInt n) alpha pa lda px ldb
  where
  n' = if side == SideLeft then natInt m else natInt n

-- | Matrix-vector multiplication.
-- XXX: Give this a more specific type
gemv :: CNum a => a -> (MatP t m n a, Operation) -> VecP u nx a 
     -> a -> VecP s ny a -> RW s ()
gemv alpha (MatP pa (m,n) lda, transa) (VecP px nx incx) 
     beta                              (VecP py ny incy) = 
  dimCheck "gemv" [[natInt nx, n'], [natInt ny, m']] . RW $ 
    C.gemv handle transa mi ni alpha pa lda px incx beta py incy
  where 
  mi = natInt m
  ni = natInt n
  (m', n') = if transa == N then (mi, ni) else (ni, mi)

-- | Matrix-matrix multiplication.
-- XXX : Give this a more specific type
gemm :: CNum a => a -> (MatP t ma na a, Operation) -> (MatP u mb nb a, Operation) 
     -> a -> MatP s m' n' a -> RW s ()
gemm alpha (MatP pa (ma, na) lda, transa) (MatP pb (mb, nb) ldb, transb)
      beta (MatP pc (m', n') ldc) = 
  dimCheck "gemm" [[m, natInt m'], [k, k'], [n, natInt n']] . RW $
    C.gemm handle transa transb m n k alpha pa lda pb ldb beta pc m
  where
  mai = natInt ma
  nai = natInt na
  mbi = natInt mb
  nbi = natInt nb
  (m ,k) = if transa == N then (mai, nai) else (nai, mai)
  (k',n) = if transb == N then (mbi, nbi) else (nbi, mbi)

-- | Symmetric matrix-vector multiplication.
symv :: CNum a => (MatP t n n a, FillMode) -> a -> VecP u n a 
     -> a -> VecP s n a -> RW s ()
symv (MatP pa (n,n') lda, uplo) alpha (VecP px nx incx)
     beta (VecP py ny incy) = 
  dimCheck "symv" [[n,n',nx,ny]] . RW $
    C.symv handle uplo (natInt n) alpha pa lda px incx beta py incy

-- | Symmetric low-rank addition.
-- XXX: Give this a more specific type
syrk :: CNum a => (MatP t ma na a, Operation) -> a -> a 
     -> (MatP s mb nb a, FillMode) -> RW s ()
syrk (MatP pa (ma,na) lda, transa) alpha beta
     (MatP pc (n ,n') ldc, uplo) =
  dimCheck "syrk" [[ni,natInt n'], [natInt na, na']] . RW $ 
    C.syrk handle uplo transa ni k alpha pa lda beta pc ldc
  where
  ni = natInt n
  k = if transa == N then natInt ma else natInt na
  na' = if transa == N then k else ni

-- | Vector dot product.
dot :: CNum a => VecP t n a -> VecP u n a -> RW s a
dot (VecP px nx incx) (VecP py ny incy) = dimCheck "dot" [[nx,ny]] . RW $
  C.dot handle (natInt nx) px incx py incy

-- | Symmetric rank-1 addition.
ger :: CNum a => a -> VecP t m a -> VecP u n a -> MatP s m n a -> RW s ()
ger alpha (VecP px mx incx) (VecP py ny incy) (MatP pa (m,n) lda) =
  dimCheck "ger" [[natInt mx, natInt m], [natInt ny, natInt n]] . RW $
    C.ger handle (natInt m) (natInt n) alpha px incx py incy pa lda

-- | Scalar multiplication for vectors.
scal :: CNum a => a -> VecP s n a -> RW s ()
scal alpha (VecP px nx incx) = RW $ C.scal handle (natInt nx) alpha px incx

-- | Combined scalar multiplication of and vector addition to a vector.
axpy :: CNum a => a -> VecP t n a -> VecP s n a -> RW s ()
axpy alpha (VecP px nx incx) (VecP py ny incy) = 
  dimCheck "axpy" [[nx,ny]] . RW $
    C.axpy handle (natInt nx) alpha px incx py incy

-- | General addition of matrices.
-- XXX : Give this a more specific type
geam :: CNum a => a -> (MatP t ma na a, Operation) 
     -> a -> (MatP u mb nb a, Operation) -> MatP s m n a -> RW s ()
geam alpha (MatP pa (ma, na) lda, transa)
     beta  (MatP pb (mb, nb) ldb, transb) (MatP pc (m, n) ldc) = 
  dimCheck "geam" [[mi, m', m''], [ni, n', n'']] . RW $
    C.geam handle transa transb mi ni alpha pa lda beta pb ldb pc ldc
  where
  mi = natInt m
  ni = natInt n
  mai = natInt ma
  nai = natInt na
  mbi = natInt mb
  nbi = natInt nb
  (m' , n' ) = if transa == N then (mai,nai) else (nai,mai)
  (m'', n'') = if transb == N then (mbi,nbi) else (nbi,mbi)

-- | Multiplication of a matrix by a diagonal matrix.
-- XXX: Give this a more specific type
dgmm :: CNum a => SideMode -> VecP t nx a -> MatP u m n a
     -> MatP s m n a -> RW s ()
dgmm mode (VecP px nx incx) (MatP pa (m ,n ) lda)
                            (MatP pc (m',n') ldc) = 
  dimCheck "dgmm" [[mi, natInt m'], [ni,natInt n'], [natInt nx, vecl]] . RW $
    C.dgmm handle mode mi ni pa lda px incx pc ldc
  where
  ni = natInt n
  mi = natInt m
  vecl = if mode == SideRight then ni else mi


-- | Make a transposed copy of a matrix.
memtrans :: CNum a => MatP t m n a -> RW s (MatP s n m a)
memtrans a@(MatP _ (m,n) _) = do
  b <- makeMatP (n,m)
  geam 1 (a, T) 0 (b, N) b
  return b

-- | Treat a matrix as one gigantic vector (throws an error if the leading
-- dimension is larger than the number of rows).
asVecP :: MatP s m n a -> VecP s (m * n) a
asVecP (MatP pa (m,n) lda) = if natInt m == lda 
  then VecP pa (times m n) 1 
  else error "asVecP: not yet defined"

-- | Produce a list of views of each of the columns of a matrix.
asColumns :: Storable a => MatP s m n a -> Vect n (VecP s m a)
asColumns (MatP pa (m,n) lda) = V.generate n $ \i ->
  VecP (advanceDevPtr pa (i * lda)) m 1

-- | Copy each vector in a list into a contiguous matrix.
fillWithColumns :: (Storable a, C.Cublas a) =>
    [VecP t m a] -> MatP s m n a -> RW s ()
fillWithColumns [] _ = return ()
fillWithColumns (vec : vs) (MatP pa (m,n) lda) = do
  copy vec (VecP pa m 1)
  fillWithColumns vs (MatP (advanceDevPtr pa lda) (m, n') lda)
  where Just n' = minus n (lit (Proxy :: Proxy 1))

-- | For each list that is provided, checks that all of the elemnts in the
-- list are the same. If not, causes an 'error' which is prefixed by the
-- provided string.
dimCheck :: (Eq b, Show b) => String -> [[b]] -> a -> a
dimCheck fname dims x = 
  case filter (not . listEq) dims of
    []       -> x
    failures -> error (fname ++ ": Dimensions not equal: " ++ show failures)
  where
  listEq [] = True
  listEq (x:xs) = all (x==) xs
