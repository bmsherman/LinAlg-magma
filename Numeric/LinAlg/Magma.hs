{-# LANGUAGE MultiParamTypeClasses, FlexibleInstances #-}
{-# LANGUAGE GADTs #-}

module Numeric.LinAlg.Magma (
  module Numeric.LinAlg,
  module Numeric.LinAlg.Magma,
  C.Matrix (..), C.Vec (..), CFloat (..), CDouble (..)
 ) where
import Numeric.LinAlg
import Foreign.C.Types (CFloat (..), CDouble (..))
import Foreign.CUDA.Cublas.Types (FillMode (..))
import qualified Numeric.LinAlg.Magma.Internal as C
import Numeric.LinAlg.Magma.Internal (GArr (..), Matrix, Vector)
import qualified Numeric.LinAlg.Magma.Mutable as C hiding (dim)
import Foreign.Storable (Storable)

instance C.CNum e => Scale e GArr where
  c .* v@(Vector _) = C.vscal c v
  c .* m@(Matrix _ _) = c C..* m

instance C.CNum e => Num (Matrix m n e) where
  a + b = a C..+ b
  a - b = a C..- b
  a * b = C.elementwiseProdM a b
  abs = error "abs"
  negate = (.*) (-1)
  signum = error "signum"
  fromInteger = error "fromInteger"

instance C.CNum e => Num (Vector n e) where
  a + b = (C.vplus a b)
  a - b = (C.vminus a b)
  a * b = (C.elementwiseProdV a b)
  abs = error "abs"
  negate = (.*) (-1)
  signum = error "signum"
  fromInteger = error "fromInteger"

instance C.CNum e => Matr e GArr where

  fromLists = C.fromLists
  toLists = C.toLists
  fromList = C.fromList
  toList = C.toList

  toRows = C.toRows
  toColumns = C.toColumns
  fromRows = C.fromRows
  fromColumns = C.fromColumns

  dim = C.mdim
  trans = C.trans
  ident = C.ident

  asColMat (Vector v) = C.asColMat v
  asColVec = Vector . C.asColVec

  fromDiag = C.fromDiag
  takeDiag = C.takeDiag'

  Vector x >.< Vector y = C.runRW $ C.dot x y
  len (Vector (C.VecP _ size _)) = size

  outer (Vector u) (Vector v) = C.outer u v

  mXm = C.mXm
  linearSolve = C.genSolveM

  chol = C.chol
  trsymprod = C.trsymprod

  elementwiseprod = C.elementwiseProdM

  ltriSolve = (C..\)
  utriSolve = (C.^\)
  posdefSolve a b = C.chol a `cholSolve` b
  l `cholSolve` b = C.trans l C.^\ (l C..\ b)


instance (C.CNum a, Eq a) => Eq (Matrix m n a) where
  x == y = C.toLists x == C.toLists y

instance (C.CNum a, Show a) => Show (Matrix m n a) where
  show = show . C.toLists

instance (C.CNum a, Show a) => Show (Vector n a) where
  show = show . C.toList
