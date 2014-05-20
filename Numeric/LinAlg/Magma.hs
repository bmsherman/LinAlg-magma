{-# LANGUAGE MultiParamTypeClasses, FlexibleInstances #-}

module Numeric.LinAlg.Magma (
  module Numeric.LinAlg,
  module Numeric.LinAlg.Magma,
  C.Matrix (..), C.Vec (..), CFloat (..), CDouble (..)
 ) where
import Numeric.LinAlg
import Foreign.C.Types (CFloat (..), CDouble (..))
import Foreign.CUDA.Cublas.Types (FillMode (..))
import qualified Numeric.LinAlg.Magma.Internal as C
import qualified Numeric.LinAlg.Magma.Mutable as C hiding (dim)
import Foreign.Storable (Storable)

instance C.CNum e => Mul e C.Matrix C.Matrix C.Matrix where
  (><) = C.mXm

instance C.CNum e => Mul e C.Matrix C.Vec C.Vec where
  (><) = C.mXv

instance C.CNum e => Mul e C.Vec C.Matrix C.Vec where
  (><) = C.vXm

instance C.CNum e => Scale e C.Vec where
  (.*) = C.vscal

instance C.CNum e => Scale e C.Matrix where
  (.*) = (C..*)

instance C.CNum e => Solve e C.Matrix C.Matrix where
  a <\> b = C.genSolveM a b
  a .\ b = a C..\ b
  a ^\ b = a C.^\ b
  a \\ b = C.chol a `cholSolve` b
  l `cholSolve` b = C.trans l ^\ (l .\ b)

instance C.CNum e => Solve e C.Matrix C.Vec where
  a <\> b = C.genSolveV a b
  a .\ b = C.tSolveV Lower a b
  a ^\ b = C.tSolveV Upper a b
  a \\ b = C.chol a `cholSolve` b
  l `cholSolve` b = C.trans l ^\ (l .\ b)

instance C.CNum e => Num (C.Matrix e) where
  a + b = a C..+ b
  a - b = a C..- b
  a * b = C.elementwiseProdM a b
  abs = error "abs"
  negate = (.*) (-1)
  signum = error "signum"
  fromInteger = error "fromInteger"

instance C.CNum e => Num (C.Vec e) where
  a + b = (C.vplus a b)
  a - b = (C.vminus a b)
  a * b = (C.elementwiseProdV a b)
  abs = error "abs"
  negate = (.*) (-1)
  signum = error "signum"
  fromInteger = error "fromInteger"

instance C.CNum e => Matr e C.Vec C.Matrix where

  fromLists = C.fromLists
  toLists = C.toLists
  fromList =  C.fromList
  toList = C.toList

  toRows = C.toRows
  toColumns = C.toColumns
  fromRows = C.fromRows
  fromColumns = C.fromColumns

  dim = C.mdim
  trans = C.trans
  ident = C.ident

  asColMat = C.asColMat
  asColVec = C.asColVec

  fromDiag = C.fromDiag
  takeDiag = C.takeDiag'

  x >.< y = C.runST $ C.dot x y
  len (C.VecP _ size _) = size

  outer = C.outer

  chol = C.chol
  trsymprod = C.trsymprod

  elementwiseprod = C.elementwiseProdM


instance (C.CNum a, Eq a) => Eq (C.Matrix a) where
  x == y = C.toLists x == C.toLists y

instance (C.CNum a, Show a) => Show (C.Matrix a) where
  show = show . C.toLists

instance (C.CNum a, Show a) => Show (C.Vec a) where
  show = show . C.toList
