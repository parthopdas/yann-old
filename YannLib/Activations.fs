module YannLib.Activations

open MathNet.Numerics.LinearAlgebra

module Sigmoid =
  let forward (Z: Matrix<double>) =
    Z.Negate().PointwiseExp().Add(1.0).PointwisePower(-1.0)

  let backward (Z: Matrix<double>) (dA: Matrix<double>) =
    let s = Z.Negate().PointwiseExp().Add(1.0).PointwisePower(-1.0) 
    dA.PointwiseMultiply(s).PointwiseMultiply(s.Negate().Add(1.0))

module ReLU =
  let forward (x: Matrix<double>) =
    x.PointwiseMaximum(0.0)

  let backward (Z: Matrix<double>) (dA: Matrix<double>) =
    dA.PointwiseMultiply(Z.PointwiseSign().PointwiseMaximum(0.0))
