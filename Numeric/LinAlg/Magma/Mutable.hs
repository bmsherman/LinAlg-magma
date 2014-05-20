{-# LANGUAGE FlexibleInstances #-}

module Numeric.LinAlg.Magma.Mutable where

import Data.List.Split (chunksOf)
import Data.Complex (Complex (..))

import Foreign.CUDA

import Foreign.C.Types (CFloat, CDouble)
import Foreign.Ptr (Ptr)
import Foreign.Storable (Storable)
import Foreign.Marshal.Array (withArray)

import Foreign.CUDA.Cublas.Types
import qualified Foreign.CUDA.Cublas.FFI as C
import qualified Foreign.CUDA.Cublas as C
import qualified Foreign.CUDA.Magma as Magma

import System.Mem.Weak (addFinalizer)
import System.IO.Unsafe (unsafePerformIO)


class (C.Cublas e, Magma.Magma e, Storable e, Floating e) => CNum e where
instance CNum CFloat
instance CNum CDouble
instance CNum (Complex CFloat)
instance CNum (Complex CDouble)

data MatP s a = MatP
  { ptrM :: DevicePtr a
  , dim  :: (Int, Int)
  , ld   :: Int }

data VecP s a = VecP
  { ptrV   :: DevicePtr a
  , len    :: Int
  , stride :: Int }

newtype ST s a = ST { unST :: IO a }

instance Monad (ST s) where
  (ST a) >>= f = ST $ a >>= unST . f
  return = ST . return

handle :: Handle
handle = unsafePerformIO $ do
  h <- C.create
  putStrLn "CUBLAS initialized."
  addFinalizer handle (C.destroy h >> putStrLn "CUBLAS finalized")
  return h

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

makeArray :: Storable a => Int -> IO (DevicePtr a)
makeArray n = mkFinalized (mallocArray n) free

makeCopyArray :: Storable a => Int -> DevicePtr a -> IO (DevicePtr a)
makeCopyArray n pa = do
  pb <- makeArray n
  copyArray n pa pb
  return pb

makeMatP :: Storable a => (Int, Int) -> ST s (MatP s a)
makeMatP dim@(m,n) = do
  p <- ST $ makeArray (m*n)
  return $ MatP p dim m

makeCopyMatP :: Storable a => MatP t a -> ST s (MatP s a)
makeCopyMatP (MatP pa dim@(m,n) lda) = 
  if m == lda 
    then do
      mat@(MatP pb _ _) <- makeMatP dim
      ST $ copyArray (m*n) pa pb
      return mat
    else error "makeCopyMatP: Not yet implemented"


copy :: C.Cublas a => VecP t a -> VecP s a -> ST s ()
copy (VecP px nx incx) (VecP py ny incy) = dimCheck "copy" [[nx,ny]] $
  ST $ C.copy handle nx px incx py incy

fromListP :: Storable a => [a] -> ST s (VecP s a)
fromListP xs = do
  px <- ST $ makeArray nx
  ST $ pokeListArray xs px
  return $ VecP px nx 1
  where nx = length xs

--Loads in COLUMN-MAJOR FORMAT!!!
fromListsP :: Storable a => [[a]] -> ST s (MatP s a)
fromListsP xs = do
  pa <- ST $ makeArray ntot
  ST $ pokeListArray (concat xs) pa
  return $ MatP pa (rows, cols) rows
  where
  ntot = rows*cols
  cols = length xs
  rows = let ls = map length xs in 
    if and (map (head ls==) ls)
      then (head ls) 
      else error "fromLists: Not all rows have the same length"

--Reads to Column major format
toListsP :: Storable a => MatP t a -> ST s [[a]]
toListsP c@(MatP pa (m,n) lda) = 
  if m == lda 
    then ST . fmap (chunksOf m) $ peekListArray (m*n) pa
    else error "toListsP: Not yet implemented"

toListP :: CNum e => VecP t e -> ST s [e]
toListP (VecP px nx 1) = ST $ peekListArray nx px
toListP v = makeCopyVecP v >>= toListP

takeDiagP :: MatP s e -> VecP s e
takeDiagP a@(MatP pa (ma,na) lda) = let n' = min ma na in 
  VecP pa n' (lda + 1)

makeVecP :: Storable a => Int -> ST s (VecP s a)
makeVecP n = do
  p <- ST $ makeArray n
  return (VecP p n 1)

setZeroMatP :: CNum a => MatP s a -> ST s ()
setZeroMatP a = geam 0 (a, N) 0 (a, N) a

makeCopyVecP :: (Storable a, C.Cublas a) => VecP t a -> ST s (VecP s a)
makeCopyVecP (VecP px nx incx) = do
  py <- ST $ makeArray nx
  ST $ C.copy handle nx px incx py 1
  return $ VecP py nx 1

potrf :: CNum a => (MatP s a, FillMode) -> ST s ()
potrf (MatP pa (n,n') lda, fill) = dimCheck "potrf" [[n, n']] $ do
  (_,res) <- magmaHandle `seq` ST $ Magma.potrf (toChar fill) n pa lda
  return $ if res /= 0 then error "Mutable.potrf: MAGMA potrf failed" else ()
  where
  toChar Upper = 'U'
  toChar Lower = 'L'

-- Warning! gesv acts destructively on both matrices
gesv :: CNum a => MatP s a -> MatP s a -> ST s ()
gesv (MatP pa (n,n') lda) (MatP pb (n'', m) ldb) = 
  dimCheck "gesv" [[n, n', n'']] $ do
    (_,res) <- magmaHandle `seq` ST $ withArray (replicate (min m n) 0) $ \ipiv -> 
      Magma.gesv n m pa lda ipiv pb ldb
    return $ if res /= 0 then error "Mutable.gesv: MAGMA gesv failed" else ()

trsv :: CNum a => (MatP t a, FillMode, Operation, DiagType)
     -> VecP s a -> ST s ()
trsv (MatP pa (n,n') lda, uplo, transa, diag) (VecP px nx incx) = 
  dimCheck "trsv" [[n, n', nx]] . ST $
    C.trsv handle uplo transa diag n pa lda px incx

trsm :: CNum a => (MatP t a, SideMode, FillMode, Operation, DiagType)
     -> a -> MatP s a -> ST s ()
trsm (MatP pa (ma,na) lda, side, uplo, transa, diag) alpha
     (MatP px (m, n) ldb) = 
  dimCheck "trsm" [[ma,na,n']] . ST $
    C.trsm handle side uplo transa diag m n alpha pa lda px ldb
  where
  n' = if side == SideLeft then m else n

gemv :: CNum a => a -> (MatP t a, Operation) -> VecP u a 
     -> a -> VecP s a -> ST s ()
gemv alpha (MatP pa (m,n) lda, transa) (VecP px nx incx) 
     beta                              (VecP py ny incy) = 
  dimCheck "gemv" [[nx, n'], [ny, m']] . ST $ 
    C.gemv handle transa m n alpha pa lda px incx beta py incy
  where (m', n') = if transa == N then (m, n) else (n, m)

gemm :: CNum a => a -> (MatP t a, Operation) -> (MatP u a, Operation) 
     -> a -> MatP s a -> ST s ()
gemm alpha (MatP pa (ma, na) lda, transa) (MatP pb (mb, nb) ldb, transb)
      beta (MatP pc (m', n') ldc) = 
  dimCheck "gemm" [[m, m'], [k, k'], [n, n']] . ST $
    C.gemm handle transa transb m n k alpha pa lda pb ldb beta pc m
  where
  (m ,k) = if transa == N then (ma, na) else (na, ma)
  (k',n) = if transb == N then (mb, nb) else (nb, mb)

symv :: CNum a => (MatP t a, FillMode) -> a -> VecP u a 
     -> a -> VecP s a -> ST s ()
symv (MatP pa (n,n') lda, uplo) alpha (VecP px nx incx)
     beta (VecP py ny incy) = 
  dimCheck "symv" [[n,n',nx,ny]] . ST $
    C.symv handle uplo n alpha pa lda px incx beta py incy

syrk :: CNum a => (MatP t a, Operation) -> a -> a 
     -> (MatP s a, FillMode) -> ST s ()
syrk (MatP pa (ma,na) lda, transa) alpha beta
     (MatP pc (n ,n') ldc, uplo) =
  dimCheck "syrk" [[n,n'], [na, na']] . ST $ 
    C.syrk handle uplo transa n k alpha pa lda beta pc ldc
  where
  k = if transa == N then ma else na
  na' = if transa == N then k else n

dot :: CNum a => VecP t a -> VecP u a -> ST s a
dot (VecP px nx incx) (VecP py ny incy) = dimCheck "dot" [[nx,ny]] . ST $
  C.dot handle nx px incx py incy
    
ger :: CNum a => a -> VecP t a -> VecP u a -> MatP s a -> ST s ()
ger alpha (VecP px mx incx) (VecP py ny incy) (MatP pa (m,n) lda) =
  dimCheck "ger" [[mx, m], [ny, n]] . ST $
    C.ger handle m n alpha px incx py incy pa lda

scal :: CNum a => a -> VecP s a -> ST s ()
scal alpha (VecP px nx incx) = ST $ C.scal handle nx alpha px incx

axpy :: CNum a => a -> VecP t a -> VecP s a -> ST s ()
axpy alpha (VecP px nx incx) (VecP py ny incy) = 
  dimCheck "axpy" [[nx,ny]] . ST $
    C.axpy handle nx alpha px incx py incy

geam :: CNum a => a -> (MatP t a, Operation) 
     -> a -> (MatP u a, Operation) -> MatP s a -> ST s ()
geam alpha (MatP pa (ma, na) lda, transa)
     beta  (MatP pb (mb, nb) ldb, transb) (MatP pc (m, n) ldc) = 
  dimCheck "geam" [[m, m', m''], [n, n', n'']] . ST $
    C.geam handle transa transb m n alpha pa lda beta pb ldb pc ldc
  where
  (m' , n' ) = if transa == N then (ma,na) else (na,ma)
  (m'', n'') = if transb == N then (mb,nb) else (nb,mb)

dgmm :: CNum a => SideMode -> VecP t a -> MatP u a
     -> MatP s a -> ST s ()
dgmm mode (VecP px nx incx) (MatP pa (m ,n ) lda)
                            (MatP pc (m',n') ldc) = 
  dimCheck "dgmm" [[m, m'], [n,n'], [nx, vecl]] . ST $
    C.dgmm handle mode m n pa lda px incx pc ldc
  where
  vecl = if mode == SideRight then n else m

memtrans :: CNum a => MatP t a -> ST s (MatP s a)
memtrans a@(MatP _ (m,n) _) = do
  b <- makeMatP (n,m)
  geam 1 (a, T) 0 (b, N) b
  return b

asVecP :: MatP s a -> VecP s a
asVecP (MatP pa (m,n) lda) = if m == lda 
  then VecP pa (m * n) 1 
  else error "asVecP: not yet defined"

asColumns :: Storable a => MatP s a -> [VecP s a]
asColumns (MatP pa (m,n) lda) = [ VecP (advanceDevPtr pa (i * lda)) m 1 | i <- [0 .. n - 1] ]

fillWithColumns :: (Storable a, C.Cublas a) => [VecP t a] -> MatP s a -> ST s ()
fillWithColumns [] _ = return ()
fillWithColumns (vec : vs) (MatP pa (m,n) lda) = do
  copy vec (VecP pa m 1)
  fillWithColumns vs (MatP (advanceDevPtr pa lda) (m, n-1) lda)

dimCheck :: (Eq b, Show b) => String -> [[b]] -> a -> a
dimCheck fname dims x = 
  case filter (not . listEq) dims of
    []       -> x
    failures -> error (fname ++ ": Dimensions not equal: " ++ show failures)
  where
  listEq [] = True
  listEq (x:xs) = all (x==) xs
