module YannLib.Core

open MathNet.Numerics.LinearAlgebra
open System.Collections.Generic

let _invalidMatrix = Matrix<double>.Build.Sparse(1, 1, 0.0)
let _invalidVector = Vector<double>.Build.Sparse(1, 0.0)

type Activation =
  | ReLU
  | Sigmoid

type Layer =
  { n: int
    Activation: Activation }

type Architecture =
  { nₓ: int
    Layers: Layer list }

type Gradient = { dA: Matrix<double>; dW: Matrix<double>; db: Vector<double> }
type Gradients = Map<int, Gradient>

type Cache = { A: Matrix<double>; Z: Matrix<double>; W: Matrix<double>; b: Vector<double> }
type Caches = Map<int, Cache>
let _invalidCache = { Z = _invalidMatrix; A = _invalidMatrix; W = _invalidMatrix; b = _invalidVector }

type Network =
  { Architecture: Architecture
    Parameters: IDictionary<int, {| W: Matrix<double>; b: Vector<double> |}> }

let _initializeNetwork (seed: int) (arch: Architecture): Network =
  let ws =
    seq { yield arch.nₓ; yield! arch.Layers |> Seq.map (fun l -> l.n) }
    |> Seq.pairwise
    |> Seq.map (fun (nPrev, n) -> Matrix<double>.Build.Random(n, nPrev, seed) * 0.01)

  let bs =
    arch.Layers
    |> Seq.map (fun l -> Vector<double>.Build.Random(l.n, seed) * 0.01)

  let ps =
    Seq.zip ws bs
    |> Seq.mapi (fun i (W, b) -> i + 1, {| W = W; b = b |})
    |> dict

  { Architecture = arch; Parameters = ps }

let _linearForward (A: Matrix<double>) (W: Matrix<double>) (b: Vector<double>) =
  let Z = W.Multiply(A) + b.BroadcastC(A.ColumnCount)
  Z, (W, b)

let _linearActivationForward (Aprev: Matrix<double>) (W: Matrix<double>) (b: Vector<double>) activation =
  let Z, linearCache = _linearForward Aprev W b

  let A, activationCache =
    match activation with
    | ReLU -> Z.PointwiseMaximum(0.0), Z
    | Sigmoid -> Z.Negate().PointwiseExp().Add(1.0).PointwisePower(-1.0), Z

  A, (linearCache, activationCache)

let _forwardPropagate network (X: Matrix<double>): (Matrix<double> * Caches) =
  let _folder =
    fun (acc: Map<int, Cache>) (l, layer: Layer) ->
      let APrev = acc.[l - 1].A
      let (A, ((W, b), Z)) = _linearActivationForward APrev network.Parameters.[l].W network.Parameters.[l].b layer.Activation
      acc |> Map.add l { A = A; Z = Z; W = W; b = b }

  let c0 = [(0, { _invalidCache with A = X })] |> Map.ofList
  let caches =
    network.Architecture.Layers
    |> List.mapi (fun i l -> (i + 1), l)
    |> List.fold _folder c0
    |> Map.remove 0

  let AL = caches.[network.Architecture.Layers.Length].A
  AL, caches

let _computeCost (Y: Matrix<double>) (Ŷ: Matrix<double>): double =
  let m = double Y.ColumnCount

  let cost = Y.Multiply(Ŷ.PointwiseLog().Transpose()) + (Y.Negate().Add(1.0).Multiply(Ŷ.Negate().Add(1.0).PointwiseLog().Transpose()))
  assert ((cost.RowCount, cost.ColumnCount) = (1, 1))

  (-1.0 / m) * cost.Item(0, 0)

let _computeAccuracy (Y: Matrix<double>) (Ŷ: Matrix<double>) =
  Prelude.undefined

let _linearBackward (dZ: Matrix<double>) { A = Aprev; W = W } =
  let m = Aprev.Shape().[1] |> double

  let dW = (1.0 / m) * dZ.Multiply(Aprev.Transpose())
  let db = (1.0 / m) * (dZ.EnumerateColumns() |> Seq.reduce (+))
  let dAprev = W.Transpose().Multiply(dZ)
  
  dAprev, dW, db

let _linearActivationBackward (dA: Matrix<double>) cache activation =
  let dZ =
    match activation with
    | ReLU ->
      let dZ = dA.Clone()
      dZ.MapIndexedInplace(fun r c v -> if cache.Z.Item(r, c) <= 0.0 then 0.0 else v)
      dZ
    | Sigmoid ->
      let s = cache.Z.Negate().PointwiseExp().Add(1.0).PointwisePower(-1.0)
      dA.PointwiseMultiply(s).PointwiseMultiply(s.Negate().Add(1.0))

  _linearBackward dZ cache

let _backwardPropagate (network: Network) (AL: Matrix<double>) (Y: Matrix<double>) (caches: Caches): Gradients =
  Prelude.undefined

let _updateParameters (network: Network) (lr: double) (gradients: Gradients) =
  let updateLayerParameters l =
    let W = network.Parameters.[l].W - lr * gradients.[l].dW
    let b = network.Parameters.[l].b - lr * gradients.[l].db
    network.Parameters.[l] <- {| W = W; b = b |}
  for i = 1 to network.Architecture.Layers.Length do
    updateLayerParameters i

let trainNetwork (seed: int) (callback: double -> double -> unit) (arch: Architecture) (X: Matrix<double>) (Y: Matrix<double>) (lr: double) (epochs: int): Network =
  let network = arch |> _initializeNetwork seed

  //for _ = 0 to (epochs - 1) do
  //  let Ŷ = _forwardPropagate network X

  //  let J = _computeCost Y Ŷ
  //  let accuracy = _computeAccuracy Y Ŷ

  //  let gradients = _backwardPropagate network Y Ŷ

  //  _updateParameters network lr gradients

  //  callback J accuracy

  network

(*
TODO
- Refactor out relu/sigmoid and their backwards
- linear cache and actiavation cache seperation
- consolidate caches
- unit tests are independent of activation functions


- compare perf with numpy
- gradient checking

*)

