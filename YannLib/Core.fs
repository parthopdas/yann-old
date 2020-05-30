module YannLib.Core

open MathNet.Numerics.LinearAlgebra
open System.Collections.Generic

type Activation =
  | ReLU
  | Sigmoid

type Layer =
  | FullyConnected of {| n: int; Activation: Activation |}

type Architecture =
  { n_x: int
    Layers: Layer array }

type Gradients = IReadOnlyDictionary<int, {| dW: Matrix<double>; db: Matrix<double> |}>

type Network =
  { Architecture: Architecture
    Parameters: IDictionary<int, {| W: Matrix<double>; b: Matrix<double> |}> }

let _initializeNetwork (seed: int) (arch: Architecture): Network =
  let ws =
    seq {
      yield arch.n_x
      yield! arch.Layers |> Seq.map (function | FullyConnected fc -> fc.n)
    }
    |> Seq.pairwise
    |> Seq.map (fun (nprev, ncurr) -> Matrix<double>.Build.Random(ncurr, nprev, seed) * 0.01)

  let bs =
    arch.Layers
    |> Seq.map (function | FullyConnected fc -> fc.n)
    |> Seq.map (fun n -> Matrix<double>.Build.Random(n, 1, seed) * 0.01)

  let ps =
    Seq.zip ws bs
    |> Seq.mapi (fun i (W, b) -> i + 1, {| W = W; b = b |})
    |> dict

  { Architecture = arch; Parameters = ps }

let _computeCost (Y: Matrix<double>) (Ŷ: Matrix<double>): double =
  let m = double Y.ColumnCount

  let cost = Y.Multiply(Ŷ.PointwiseLog().Transpose()) + (Y.Negate().Add(1.0).Multiply(Ŷ.Negate().Add(1.0).PointwiseLog().Transpose()))
  assert ((cost.RowCount, cost.ColumnCount) = (1, 1))

  (-1.0 / m) * cost.Item(0, 0)

let _computeAccuracy (Y: Matrix<double>) (Ŷ: Matrix<double>) =
  Prelude.undefined

let _forwardPropagate network X =
  Prelude.undefined

let _backwardPropagate (network: Network) (Y: Matrix<double>) (Ŷ: Matrix<double>): Gradients =
  Prelude.undefined

let _updateParameters (network: Network) (lr: double) (gradients: Gradients) =
  let updateLayerParameters i _ =
    let l = i + 1
    let W = network.Parameters.[l].W - lr * gradients.[l].dW
    let b = network.Parameters.[l].b - lr * gradients.[l].db
    network.Parameters.[l] <- {| W = W; b = b |}
  network.Architecture.Layers |> Seq.iteri updateLayerParameters

let trainNetwork (seed: int) (callback: double -> double -> unit) (arch: Architecture) (X: Matrix<double>) (Y: Matrix<double>) (lr: double) (epochs: int): Network =
  let network = arch |> _initializeNetwork seed

  for _ = 0 to (epochs - 1) do
    let Ŷ = _forwardPropagate network X

    let J = _computeCost Y Ŷ
    let accuracy = _computeAccuracy Y Ŷ

    let gradients = _backwardPropagate network Y Ŷ

    _updateParameters network lr gradients

    callback J accuracy

  network
