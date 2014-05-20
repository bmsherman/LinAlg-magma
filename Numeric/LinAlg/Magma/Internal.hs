{-# LANGUAGE RankNTypes #-}

module Numeric.LinAlg.Magma.Internal where

import Control.Applicative ((<$>))
import Control.Monad (forM_)

import Data.List (foldl', transpose)
import Data.List.Split (chunksOf)

import Foreign.CUDA
import Foreign.C.Types (CFloat, CInt (CInt))
import Foreign.Ptr (Ptr)
import Foreign.Storable (Storable)

import Foreign.CUDA.Cublas.Types
import qualified Foreign.CUDA.Magma as Magma

import System.IO.Unsafe (unsafePerformIO)

import Numeric.LinAlg.Magma.Mutable hiding (dim, ld)

data Ice

type Mat = MatP Ice
type Vec = VecP Ice

makeST :: (forall s. ST s (c s e)) -> c Ice e
makeST (ST x) = unsafePerformIO x

runST :: (forall s. ST s a) -> a
runST (ST x) = unsafePerformIO x


data Matrix a = Matrix {
  payload :: Mat a, 
  trns :: Bool
  }

trans :: Matrix a -> Matrix a
trans (Matrix p t) = Matrix p (not t)

trOp :: Matrix a -> Operation
trOp = transOp . trns

transOp :: Bool -> Operation
transOp True = T
transOp False = N

mdim :: Matrix a -> (Int, Int)
mdim (Matrix (MatP _ (m,n) _) t) = if t then (n,m) else (m,n)

mrows, mcols :: Matrix a -> Int
mrows = fst . mdim
mcols = snd . mdim

creating :: (forall s. ST s (c s e)) -> (forall s. c s e -> ST s b) 
         -> c Ice e
creating creator f = makeST $ do
  x <- creator
  f x
  return x

fromList xs = makeST $ fromListP xs
toList x = runST $ toListP x

--accepts input in ROW-MAJOR format, for now
fromLists :: Storable a => [[a]] -> Matrix a
fromLists xs = Matrix (makeST $ fromListsP (transpose xs)) False

toLists :: CNum a => Matrix a -> [[a]]
toLists a@(Matrix pa ta) = if ta then runST $ toListsP pa else toLists (memtrans' a)

mXm :: CNum e => Matrix e -> Matrix e -> Matrix e
mXm a@(Matrix pa t) b@(Matrix pb t') = dimCheck "mXm" [[k,k']] $ 
  Matrix p False
  where
  p = creating (makeMatP (m,n)) $ \pc ->
    gemm 1 (pa, transOp t) (pb, transOp t') 0 pc
  (m ,k) = mdim a
  (k',n) = mdim b

mXv :: CNum e => Matrix e -> Vec e -> Vec e
mXv a@(Matrix pa t) x = dimCheck "mXv" [[n, len x]] $
  creating (makeVecP m) $ \y ->
    gemv 1 (pa, transOp t) x 0 y
  where
  (m,n) = mdim a

vXm x a = mXv (trans a) x

(.+), (.-) :: CNum e => Matrix e -> Matrix e -> Matrix e
(.+) = addScale 1
(.-) = addScale (-1)

(.*) :: CNum e => e -> Matrix e -> Matrix e
k .* a@(Matrix pa t) = Matrix pb t where
  pb = creating (makeCopyMatP pa) $ \p ->
    scal k (asVecP p)

trsymprod :: CNum e => Matrix e -> Matrix e -> e
trsymprod a@(Matrix pa ta) b@(Matrix pb tb) =
  if ta==tb 
    then runST $ dot (asVecP pa) (asVecP pb)
    else trsymprod a (memtrans' b)


--Switch between row-major and column-major storage
memtrans' :: CNum e => Matrix e -> Matrix e
memtrans' a@(Matrix pa ta) = Matrix pa' (not ta) where
  pa' = makeST $ memtrans pa 


-- | addScale beta a b = a + beta*b
addScale :: CNum e => e -> Matrix e -> Matrix e -> Matrix e
addScale beta a@(Matrix pa ta) b@(Matrix pb tb) = Matrix pc False where
  pc = creating (makeMatP (mdim a)) $ \p ->
    geam 1 (pa, transOp ta) beta (pb, transOp tb) p

genSolveV :: CNum e => Matrix e -> Vec e -> Vec e
genSolveV a@(Matrix pa tra) b = 
  creating (makeCopyVecP b) $ \(VecP px n 1) -> do
    pa' <- makeCopy tra pa
    gesv pa' (MatP px (n, 1) n)
  where
  makeCopy trbool = if trbool then memtrans else makeCopyMatP

genSolveM :: CNum e => Matrix e -> Matrix e -> Matrix e
genSolveM a@(Matrix pa tra) b@(Matrix pb trb) = Matrix px False
  where
  makeCopy trbool = if trbool then memtrans else makeCopyMatP
  px = creating (makeCopy trb pb) $ \pb -> do
    pa' <- makeCopy tra pa
    gesv pa' pb
  
tSolveM :: CNum e => FillMode -> SideMode -> Matrix e -> Matrix e -> Matrix e
tSolveM fill side u b@(Matrix _ True) = 
  trans $ tSolveM (flip fill) (swap side) (trans u) (trans b) where
  flip Upper = Lower
  flip Lower = Upper
  swap SideLeft = SideRight
  swap SideRight = SideLeft
tSolveM fill side a@(Matrix pa ta) b@(Matrix pb False) = Matrix px False where
  px = creating (makeCopyMatP pb) $ \p ->
    trsm (pa, side, actfill fill, transOp ta, NonUnit) 1 p
  flip Upper = Lower
  flip Lower = Upper
  actfill = if ta then flip else id

tSolveV :: CNum e => FillMode -> Matrix e -> Vec e -> Vec e
tSolveV fill a@(Matrix pa ta) b = creating (makeCopyVecP b) $ \x ->
  trsv (pa, actfill fill, transOp ta, NonUnit) x
  where
  flip Upper = Lower
  flip Lower = Upper
  actfill = if ta then flip else id

--IMPORTANT NOTE: THE RETURNED MATRIX IS CORRECT ON THE LOWER TRIANGLE,
--BUT THE OTHER ENTRIES ARE NOT ZEROED OUT! SO BE VERY, VERY CAREFUL!
chol :: CNum e => Matrix e -> Matrix e
chol a@(Matrix pa ta) = magmaHandle `seq` Matrix pl ta where
  pl = creating (makeCopyMatP pa) $ \p -> 
    potrf (p, fill)
  fill = if ta then Upper else Lower

(^\), (.\) :: CNum e => Matrix e -> Matrix e -> Matrix e
(^\) = tSolveM Upper SideLeft
(.\) = tSolveM Lower SideLeft

vminus, vplus :: CNum e => Vec e -> Vec e -> Vec e
vminus = flip $ vAddScale (-1) 
vplus = flip $ vAddScale 1

asColMat :: CNum e => Vec e -> Matrix e
asColMat v = Matrix (asColMatP v) False

asColVec :: CNum e => Matrix e -> Vec e
asColVec a@(Matrix pa ta) = if ta
  then asColVec (memtrans' a)
  else asColVecP pa

asColMatP :: CNum e => Vec e -> Mat e
asColMatP (VecP px n 1) = MatP px (n,1) n
asColMatP v = asColMatP $ makeST (makeCopyVecP v)

asColVecP :: CNum e => Mat e -> Vec e
asColVecP a@(MatP px (m,n) lda) = if m == lda
  then VecP px (m*n) 1
  else asColVecP $ makeST (makeCopyMatP a)

outer :: CNum e => Vec e -> Vec e -> Matrix e
outer u v = Matrix pa False where
  pa = creating (makeMatP (len u, len v)) $ \p -> do
    setZeroMatP p
    ger 1 u v p

vscal :: CNum e => e -> Vec e -> Vec e
vscal alpha v = creating (makeCopyVecP v) $ \y -> 
  scal alpha y

vAddScale :: CNum e => e -> Vec e -> Vec e -> Vec e
vAddScale alpha x y = creating (makeCopyVecP y) $ \z ->
  axpy alpha x z

elementwiseProdV :: CNum e => Vec e -> Vec e -> Vec e
elementwiseProdV (VecP px nx incx) y@(VecP py ny incy) = 
  dimCheck "elementwiseProdV" [[nx, ny]] $
   creating (makeVecP nx) $ \z@(VecP pz nz incz) ->
     dgmm SideRight y (MatP px (1,nx) incx) (MatP pz (1,nz) incz)

elementwiseProdM :: CNum e => Matrix e -> Matrix e -> Matrix e
elementwiseProdM a@(Matrix pa ta) b@(Matrix pb tb) = 
  dimCheck "elementwiseProdM" [[mdim a, mdim b]] $
    if ta == tb
      then 
        let (m,n) = mdim b
            [a', b'] = map asColVec [a, b]
            (VecP pc _ 1) = elementwiseProdV a' b'
        in (Matrix (MatP pc (m,n) m) False)
      else elementwiseProdM a (memtrans' b)

ident :: CNum e => Int -> Matrix e
ident n = fromDiag (constant 1 n)

fromDiag :: CNum e => Vec e -> Matrix e
fromDiag v = Matrix payload False where
  payload = creating (makeMatP (n,n)) $ \a@(MatP pa _ lda) -> do
    setZeroMatP a
    copy v (VecP pa n (lda + 1))
  n = len v

takeDiag' :: CNum e => Matrix e -> Vec e
takeDiag' (Matrix pa _) = takeDiagP pa

toColumns :: CNum e => Matrix e -> [Vec e]
toColumns mat@(Matrix payload trans) = if trans
  then toColumns (memtrans' mat)
  else asColumns payload

toRows :: CNum e => Matrix e -> [Vec e]
toRows = toColumns . trans

fromColumns :: CNum e => [Vec e] -> Matrix e
fromColumns vs@(v : _) = Matrix payload False where
  payload = creating (makeMatP (m,n)) $ fillWithColumns vs
  m = len v
  n = length vs

fromRows :: CNum e => [Vec e] -> Matrix e
fromRows = trans . fromColumns

constant :: CNum a => a -> Int -> Vec a
constant alpha n = makeST $ do
  (VecP px _ _) <- fromListP [alpha]
  return (VecP px n 0)

constantM :: CNum a => a -> (Int, Int) -> Matrix a
constantM alpha (m,n) = Matrix (MatP pa (m,n) 0) False where
  (VecP pa _ 1) = makeST $ makeCopyVecP (constant alpha m)
