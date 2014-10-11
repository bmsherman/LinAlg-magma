{-# LANGUAGE DataKinds, KindSignatures, GADTs, TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE Rank2Types #-}

{- | This module uses the mutable interface to provide an immutable,
purely functional interface for matrix computations on the GPU. Because
most of these functions are part of the LinAlg interface, there are not
documented here. -}

module Numeric.LinAlg.Magma.Internal (
  -- * Data types
  GArr (..), Matrix, Vector, 
  SBool (..), Flip (..),

  -- * LinAlg operations
  -- ** Data transfer
  fromVect, toVect, fromVects, toVects,
  toRows, toColumns, fromRows, fromColumns,
  asColMat, asColVec,
  fromDiag, takeDiag',
  -- ** Addition and scalar multiplication
  vAddScale,
  vscal, vplus, vminus,
  (.*), (.+), (.-),
  -- ** Multiplication
  mXv, vXm, mXm,
  -- ** Core operations
  mdim, mrows, mcols, len,
  trans,
  ident,
  outer,
  elementwiseProdV, elementwiseProdM,
  -- ** Solving linear systems
  genSolveV, genSolveM,
  tSolveV, tSolveM,
  (^\), (.\),
  chol,
  -- ** Other functions
  constant, constantM, trsymprod

) where

import Control.Applicative ((<$>))
import Control.Monad (forM_)

import Data.List (foldl', transpose)
import Data.List.Split (chunksOf)
import Data.Proxy (Proxy (Proxy))

import Foreign.CUDA
import Foreign.C.Types (CFloat, CInt (CInt))
import Foreign.Ptr (Ptr)
import Foreign.Storable (Storable)

import Foreign.CUDA.Cublas.Types

import GHC.TypeLits

import System.IO.Unsafe (unsafePerformIO)

import Numeric.LinAlg (Dim (..))
import Numeric.LinAlg.Magma.Mutable hiding (dim, ld)
import Numeric.LinAlg.SNat (SNat, snat, lit, times)
import qualified Numeric.LinAlg.Vect as V
import Numeric.LinAlg.Vect (Vect, cons, nil)

data SBool (b :: Bool) where 
  STrue  :: SBool True
  SFalse :: SBool False

type family Flip (t :: Bool) (m :: Nat) (n :: Nat) :: Nat where
  Flip True  m n = n
  Flip False m n = m

-- | An immutable matrix datatype with a transpose flag, so we can avoid
-- unnecessary transposition.
data GArr :: Dim -> * -> * where
  Matrix :: !(Mat m n a) -> !(SBool b) -> GArr (M (Flip b m n) (Flip b n m)) a
  Vector :: !(Vec n a) -> GArr (V n) a

type Matrix m n = GArr (M m n)
type Vector n = GArr (V n)

-- | Matrix transpose.
trans :: Matrix m n a -> Matrix n m a
trans (Matrix p STrue) = Matrix p SFalse
trans (Matrix p SFalse) = Matrix p STrue

trOp :: Matrix m n a -> Operation
trOp (Matrix _ STrue) = T
trOp (Matrix _ SFalse) = N

sbool :: SBool b -> Bool
sbool (STrue) = True
sbool (SFalse) = False

transOp :: SBool b -> Operation
transOp STrue = T
transOp SFalse = N

-- | Dimension of a matrix (rows, columns).
mdim :: Matrix m n a -> (SNat m , SNat n)
mdim (Matrix (MatP _ (m,n) _) STrue) = (n, m)
mdim (Matrix (MatP _ (m,n) _) SFalse) = (m, n)

mrows :: Matrix m n a -> SNat m
mrows = fst . mdim
mcols :: Matrix m n a -> SNat n 
mcols = snd . mdim

fromVect :: CNum e => Vect n e -> GArr (V n) e
fromVect xs = Vector (makeRWV (fromVectP xs))

toVect :: CNum e => Vector n e -> Vect n e
toVect (Vector x) = runRW $ toVectP x

-- | In line with the LinAlg specification, this function accepts
-- input in /row-major/ format.
fromVects :: Storable a => Vect m (Vect n a) -> GArr (M m n) a 
fromVects xs = Matrix (makeRWM $ fromVectsP xs) STrue

-- | Converts a matrix to a list of lists in /row-major/ format.
toVects :: CNum a => Matrix m n a -> Vect m (Vect n a)
toVects (Matrix pa STrue) = runRW $ toVectsP pa 
toVects a@(Matrix _ SFalse) = toVects (memtrans' a)

mXm :: CNum e => Matrix m n e -> Matrix n p e -> Matrix m p e
mXm a@(Matrix pa t) b@(Matrix pb t') = dimCheck "mXm" [[k,k']] $ 
  Matrix p SFalse
  where
  p = creatingM (makeMatP (m,n)) $ \pc ->
    gemm 1 (pa, transOp t) (pb, transOp t') 0 pc
  (m ,k) = mdim a
  (k',n) = mdim b

mXv :: CNum e => Matrix m n e -> Vector n e -> Vector m e
mXv a@(Matrix pa t) (Vector x) = dimCheck "mXv" [[n, len x]] . Vector $
  creatingV (makeVecP m) $ \y ->
    gemv 1 (pa, transOp t) x 0 y
  where
  (m,n) = mdim a

vXm x a = mXv (trans a) x

(.+), (.-) :: CNum e => Matrix m n e -> Matrix m n e -> Matrix m n e
(.+) = addScale 1
(.-) = addScale (-1)

(.*) :: CNum e => e -> Matrix m n e -> Matrix m n e
k .* a@(Matrix pa t) = Matrix pb t where
  pb = creatingM (makeCopyMatP pa) $ \p ->
    scal k (asVecP p)

trsymprod :: CNum e => Matrix n n e -> Matrix n n e -> e
trsymprod = sameTrans1 f where
  f pa pb = runRW $ dot (asVecP pa) (asVecP pb)

sameTrans1 :: CNum e => (forall i. forall j. Mat i j e -> Mat i j e -> a)
                     -> Matrix m n e -> Matrix m n e -> a
sameTrans1 f (Matrix a STrue) (Matrix b STrue) = f a b
sameTrans1 f (Matrix a SFalse) (Matrix b SFalse) = f a b
sameTrans1 f (Matrix a STrue) (Matrix b SFalse) = f (makeRWM (memtrans a)) b
sameTrans1 f (Matrix a SFalse) (Matrix b STrue) = f a (makeRWM (memtrans b))

sameTrans :: CNum e => (forall i. forall j. Mat i j e -> Mat i j e -> Mat i j e)
          ->  Matrix m n e -> Matrix m n e -> Matrix m n e
sameTrans f (Matrix pa STrue) (Matrix pb STrue) = Matrix (f pa pb) STrue
sameTrans f (Matrix pa SFalse) (Matrix pb SFalse) = Matrix (f pa pb) SFalse
sameTrans f (Matrix pa STrue) (Matrix pb SFalse) = 
  Matrix (f (makeRWM (memtrans pa)) pb) SFalse
sameTrans f (Matrix pa SFalse) (Matrix pb STrue) =
  Matrix (f pa (makeRWM (memtrans pb))) SFalse

-- | Switch between row-major and column-major storage. That is, the matrix
-- is transposed in memory, but /not/ logically.
memtrans' :: CNum e => Matrix m n e -> Matrix m n e
memtrans' (Matrix pa STrue)  = Matrix (makeRWM (memtrans pa)) SFalse
memtrans' (Matrix pa SFalse) = Matrix (makeRWM (memtrans pa)) STrue


-- | addScale beta a b = a + beta*b
addScale :: CNum e => e -> Matrix m n e -> Matrix m n e -> Matrix m n e
addScale beta a@(Matrix pa ta) b@(Matrix pb tb) = Matrix pc SFalse where
  pc = creatingM (makeMatP (mdim a)) $ \p ->
    geam 1 (pa, transOp ta) beta (pb, transOp tb) p

-- | General linear solver
genSolveV :: CNum e => Matrix n n e -> Vector n e -> Vector n e
genSolveV a (Vector b) = Vector $ 
  creatingV (makeCopyVecP b) $ \(VecP px n 1) -> do
    pa' <- makeCopy a
    gesv pa' (MatP px (n, lit (Proxy :: Proxy 1)) (snat n))


makeCopy :: CNum e => Matrix m n e -> RW s (MatP s m n e)
makeCopy (Matrix pa STrue) = memtrans pa
makeCopy (Matrix pa SFalse) = makeCopyMatP pa

-- | General linear solver
genSolveM :: CNum e => Matrix n n e -> Matrix n p e -> Matrix n p e
genSolveM a b = Matrix px SFalse
  where
  px = creatingM (makeCopy b) $ \pb -> do
    pa' <- makeCopy a
    gesv pa' pb
  
-- | Solution of a triangular system.
-- XXX: make the type more specific
tSolveM :: CNum e => FillMode -> SideMode -> Matrix n n e -> Matrix m p e -> Matrix m p e
tSolveM fill side u b@(Matrix _ STrue) = 
  trans $ tSolveM (flip fill) (swap side) (trans u) (trans b) where
  flip Upper = Lower
  flip Lower = Upper
  swap SideLeft = SideRight
  swap SideRight = SideLeft
tSolveM fill side a@(Matrix pa STrue) b@(Matrix pb SFalse) = 
  Matrix px SFalse
  where
  px = creatingM (makeCopyMatP pb) $ \p ->
    trsm (pa, side, flip fill, transOp STrue, NonUnit) 1 p
  flip Upper = Lower
  flip Lower = Upper
tSolveM fill side a@(Matrix pa SFalse) b@(Matrix pb SFalse) = Matrix px SFalse
  where
  px = creatingM (makeCopyMatP pb) $ \p ->
    trsm (pa, side, fill, transOp SFalse, NonUnit) 1 p

-- | Solution of a triangular system.
tSolveV :: CNum e => FillMode -> Matrix n n e -> Vec n e -> Vec n e
tSolveV fill a@(Matrix pa STrue) b = creatingV (makeCopyVecP b) $ \x ->
  trsv (pa, flip fill, transOp STrue, NonUnit) x
  where
  flip Upper = Lower
  flip Lower = Upper
tSolveV fill a@(Matrix pa SFalse) b = creatingV (makeCopyVecP b) $ \x ->
  trsv (pa, fill, transOp SFalse, NonUnit) x

-- | Cholesky decomposition.
-- __Important Note__: The entries above the diagonal are *not* zeroed
-- out, so be careful!
chol :: CNum e => Matrix n n e -> Matrix n n e
chol (Matrix pa STrue) = magmaHandle `seq` Matrix pl STrue where
  pl = creatingM (makeCopyMatP pa) $ \p -> 
    potrf (p, Upper)
chol (Matrix pa SFalse) = magmaHandle `seq` Matrix pl SFalse where
  pl = creatingM (makeCopyMatP pa) $ \p -> 
    potrf (p, Lower)

--fill SFalse = Lower

(^\), (.\) :: CNum e => Matrix n n e -> Matrix n p e -> Matrix n p e
(^\) = tSolveM Upper SideLeft
(.\) = tSolveM Lower SideLeft

vminus, vplus :: CNum e => Vector n e -> Vector n e -> Vector n e
vminus = flip $ vAddScale (-1) 
vplus = flip $ vAddScale 1

asColMat :: CNum e => Vec n e -> Matrix n 1 e
asColMat v = Matrix (asColMatP v) SFalse

asColVec :: CNum e => Matrix m n e -> Vec (m * n) e
asColVec a@(Matrix _ STrue) = asColVec (memtrans' a)
asColVec (Matrix pa SFalse) = asColVecP pa

asColMatP :: CNum e => Vec n e -> Mat n 1 e
asColMatP (VecP px n 1) = MatP px (n,lit (Proxy :: Proxy 1)) (snat n)
asColMatP v = asColMatP $ makeRWV (makeCopyVecP v)

asColVecP :: CNum e => Mat m n e -> Vec (m * n) e
asColVecP a@(MatP px (m,n) lda) = if snat m == lda
  then VecP px (times m n) 1
  else asColVecP $ makeRWM (makeCopyMatP a)

outer :: CNum e => Vec m e -> Vec n e -> Matrix m n e
outer u v = Matrix pa SFalse where
  pa = creatingM (makeMatP (len u, len v)) $ \p -> do
    setZeroMatP p
    ger 1 u v p

vscal :: CNum e => e -> Vector n e -> Vector n e
vscal alpha (Vector v) = Vector $ creatingV (makeCopyVecP v) $ \y -> 
  scal alpha y

vAddScale :: CNum e => e -> Vector n e -> Vector n e -> Vector n e
vAddScale alpha (Vector x) (Vector y) = Vector $
  creatingV (makeCopyVecP y) $ \z ->
    axpy alpha x z

elementwiseProdVP :: CNum e => Vec n e -> Vec n e -> Vec n e
elementwiseProdVP (VecP px nx incx) y@(VecP py ny incy) = 
  dimCheck "elementwiseProdV" [[nx, ny]] $
   creatingV (makeVecP nx) $ \z@(VecP pz nz incz) ->
     dgmm SideRight y (MatP px (lit one,nx) incx) (MatP pz (lit one,nz) incz)
  where one = Proxy :: Proxy 1


elementwiseProdV :: CNum e => Vector n e -> Vector n e -> Vector n e
elementwiseProdV (Vector u) (Vector v) = Vector (elementwiseProdVP u v)

elementwiseProdM :: CNum e => Matrix m n e -> Matrix m n e -> Matrix m n e
elementwiseProdM = sameTrans f
  where
  f pa@(MatP _ dima@(m, n) _) pb@(MatP _ dimb _) =
    dimCheck "elementwiseProdM" [[dima, dimb]] $
       let [pa', pb'] = map asColVecP [pa, pb]
	   VecP pc _ 1 = elementwiseProdVP pa' pb'
       in MatP pc (m,n) (snat m)

ident :: CNum e => SNat n -> Matrix n n e
ident n = fromDiag (constant 1 n)

fromDiag :: CNum e => Vector n e -> Matrix n n e
fromDiag (Vector v) = Matrix payload SFalse where
  payload = creatingM (makeMatP (n,n)) $ \a@(MatP pa _ lda) -> do
    setZeroMatP a
    copy v (VecP pa n (lda + 1))
  n = len v

takeDiag' :: CNum e => Matrix n n e -> Vector n e
takeDiag' (Matrix pa STrue)  = Vector $ takeDiagP pa
takeDiag' (Matrix pa SFalse) = Vector $ takeDiagP pa

toColumns :: CNum e => Matrix m n e -> Vect n (Vector m e)
toColumns mat@(Matrix payload STrue) = toColumns (memtrans' mat)
toColumns (Matrix payload SFalse) = V.map Vector $ asColumns payload

toRows :: CNum e => Matrix m n e -> Vect m (Vector n e)
toRows = toColumns . trans

fromColumns :: CNum e => Vect n (Vector m e) -> GArr (M m n) e
fromColumns vect = Matrix payload SFalse
  where
  payload = creatingM (makeMatP (m, n)) $ 
    fillWithColumns vs
  vs@(v : _) = unVector (V.toList vect)
  n = V.length vect
  m = len v
  unVector :: [Vector n e] -> [Vec n e]
  unVector [] = []
  unVector (Vector v : vs) = v : unVector vs

fromRows :: CNum e => Vect m (Vector n e) -> GArr (M m n) e
fromRows = trans . fromColumns

constant :: CNum a => a -> SNat n -> Vector n a
constant alpha n = Vector (constantP alpha n)

constantP :: CNum a => a -> SNat n -> Vec n a
constantP alpha n = makeRWV $ do
  (VecP px _ _) <- fromVectP (cons alpha nil)
  return (VecP px n 0)

constantM :: CNum a => a -> (SNat m, SNat n) -> Matrix m n a
constantM alpha (m,n) = Matrix (MatP pa (m,n) 0) SFalse where
  (VecP pa _ 1) = makeRWV $ makeCopyVecP (constantP alpha m)
