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

type LayerParameters =
  { W: Matrix<double>
    b: Matrix<double> }

type Network =
  { Parameters: IReadOnlyDictionary<int, LayerParameters> }

let initializeNetwork (seed: int) (arch: Architecture): Network =
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
    |> Seq.mapi (fun i (W, b) -> i + 1, { W = W; b = b })
    |> readOnlyDict

  { Parameters = ps }

