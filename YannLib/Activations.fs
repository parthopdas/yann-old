module YannLib.Activations

open MathNet.Numerics.LinearAlgebra

module Sigmoid =
  let f (Z: Matrix<double>) =
    Z.Negate().PointwiseExp().Add(1.0).PointwisePower(-1.0)

  let df (Z: Matrix<double>) =
    let s = Z.Negate().PointwiseExp().Add(1.0).PointwisePower(-1.0) 
    s.PointwiseMultiply(s.Negate().Add(1.0))

module ReLU =
  let f (x: Matrix<double>) =
    x.PointwiseMaximum(0.0)

  let df (Z: Matrix<double>) =
    Z.PointwiseSign().PointwiseMaximum(0.0)

module Softmax =
  let f (x: Matrix<double>) =
    let expx = x.PointwiseExp();
    expx.Divide(expx.ColumnSums().Sum() + Constants.εDivBy0Gaurd)

  let df (Z: Matrix<double>) =
    let s = f Z
    let sisj = s.TransposeThisAndMultiply(s)
    let diag = Matrix.Build.DenseOfDiagonalArray(s.AsColumnMajorArray())
    diag - sisj
