module YannLib.Core

open MathNet.Numerics.LinearAlgebra
open System.Diagnostics
open System

(*

  # Dimensional Analysis

  ## Formula
  A₀ = X
  Zₗ = Wₗ.Aₗ₋₁ + bₗ
  Aₗ = gₗ(Zₗ)

  dim(Wₗ) = nₗ x nₗ₋₁
  dim(bₗ) = nₗ x 1
  dim(Zₗ) = nₗ₋₁ x m
  dim(Aₗ) = dim(Zₗ)

  ## Example

  nₓ      = 3
  Layers  = 4, 5, 2
  m       = 7

  [  W1    A0    b1      A1   ]    [  W2    A1    b2      A2   ]    [  W3    A2    b3   =  A3   ]
  [ (4x3).(3x7)+(4x1) = (4x7) ] -> [ (5x4).(4x7)+(5x1) = (5x7) ] -> [ (2x5).(5x7)+(2x1) = (2x7) ]

 *)

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

type HyperParameters =
  { Epochs : int
    α: double }

[<DebuggerDisplay("dA = {dA.ShapeString()}, dW = {dW.ShapeString()}, db = {db.ShapeString()}")>]
type Gradient = { dA: Matrix<double>; dW: Matrix<double>; db: Vector<double> }
type Gradients = Map<int, Gradient>
let _invalidGradient = { dA = _invalidMatrix; dW = _invalidMatrix; db = _invalidVector }

[<DebuggerDisplay("Aprev = {Aprev.ShapeString()}, W = {W.ShapeString()}, b = {b.ShapeString()}, Z = {Z.ShapeString()}")>]
type Cache = { Aprev: Matrix<double>; W: Matrix<double>; b: Vector<double>; Z: Matrix<double> }
type Caches = Map<int, Cache>
let _invalidCache = { Aprev = _invalidMatrix; W = _invalidMatrix; b = _invalidVector; Z = _invalidMatrix }

[<DebuggerDisplay("W = {W.ShapeString()}, b = {b.ShapeString()}")>]
type Parameter = { W: Matrix<double>; b: Vector<double> }
type Parameters = Map<int, Parameter>

type EpochCallback = int -> TimeSpan -> double -> double -> unit

let _initializeParameters (seed: int) (arch: Architecture): Parameters =
  let ws =
    seq { yield arch.nₓ; yield! arch.Layers |> Seq.map (fun l -> l.n) }
    |> Seq.pairwise
    |> Seq.map (fun (nPrev, n) -> Matrix<double>.Build.Random(n, nPrev, seed) * 0.01)

  let bs =
    arch.Layers
    |> Seq.map (fun l -> Vector<double>.Build.Random(l.n, seed) * 0.01)

  Seq.zip ws bs
  |> Seq.mapi (fun i (W, b) -> i + 1, { W = W; b = b })
  |> Map.ofSeq

let _linearForward (Aprev: Matrix<double>) (W: Matrix<double>) (b: Vector<double>) =
  let Z = W.Multiply(Aprev) + b.BroadcastC(Aprev.ColumnCount)
  { Aprev = Aprev; W = W; b = b; Z = Z }

let _linearActivationForward (Aprev: Matrix<double>) (W: Matrix<double>) (b: Vector<double>) activation =
  let cache = _linearForward Aprev W b

  let A =
    match activation with
    | ReLU -> cache.Z.PointwiseMaximum(0.0)
    | Sigmoid -> cache.Z.Negate().PointwiseExp().Add(1.0).PointwisePower(-1.0)

  A, cache

let _forwardPropagate arch (parameters: Parameters) (X: Matrix<double>): (Matrix<double> * Caches) =
  let _folder (APrev: Matrix<double>, acc: Map<int, Cache>) (l, layer: Layer) =
    let (A, cache) = _linearActivationForward APrev parameters.[l].W parameters.[l].b layer.Activation
    (A, acc |> Map.add l cache)

  arch.Layers
  |> List.mapi (fun i l -> (i + 1), l)
  |> List.fold _folder (X, Map.empty)

let _computeCost (Y: Matrix<double>) (Ŷ: Matrix<double>): double =
  let cost = Y.Multiply(Ŷ.PointwiseLog().Transpose()) + (Y.Negate().Add(1.0).Multiply(Ŷ.Negate().Add(1.0).PointwiseLog().Transpose()))

  let m = double Y.ColumnCount
  (-1.0 / m) * cost.Item(0, 0)

let _computeAccuracy (Y: Matrix<double>) (Ŷ: Matrix<double>) =
  0.0

let _linearBackward (dZ: Matrix<double>) { Aprev = Aprev; W = W } =
  let m = Aprev.Shape().[1] |> double

  let dW = (1.0 / m) * dZ.Multiply(Aprev.Transpose())
  let db = (1.0 / m) * (dZ.EnumerateColumns() |> Seq.reduce (+))
  let dAprev = W.Transpose().Multiply(dZ)
  
  dAprev, dW, db

let _linearActivationBackward (dA: Matrix<double>) cache activation =
  let dZ =
    match activation with
    | ReLU ->
      dA.PointwiseMultiply(cache.Z.PointwiseSign().PointwiseMaximum(0.0))
    | Sigmoid ->
      let s = cache.Z.Negate().PointwiseExp().Add(1.0).PointwisePower(-1.0)
      dA.PointwiseMultiply(s).PointwiseMultiply(s.Negate().Add(1.0))

  _linearBackward dZ cache

let _backwardPropagate arch (Y: Matrix<double>) (Ŷ: Matrix<double>) (caches: Caches): Gradients =
  let _folder (acc: Gradients) (l, layer: Layer) =
    let (dAprev, dW, db) = _linearActivationBackward acc.[l].dA caches.[l] layer.Activation
    let gradPrev = { _invalidGradient with dA = dAprev }
    let grad = { acc.[l] with dW = dW; db = db }
    acc |> Map.remove l |> Map.add l grad |> Map.add (l - 1) gradPrev

  let L = arch.Layers.Length
  let dAL = - (Y.PointwiseDivide(Ŷ) - Y.Negate().Add(1.0).PointwiseDivide(Ŷ.Negate().Add(1.0)))
  let g0 = Map.empty |> Map.add L { _invalidGradient with dA = dAL }
  let grads =
    arch.Layers
    |> List.mapi (fun i l -> (i + 1), l)
    |> List.rev
    |> List.fold _folder g0

  grads

let _updateParameters arch (α: double) (parameters: Parameters) (gradients: Gradients): Parameters =
  let _folder acc l =
    let W = parameters.[l].W - α * gradients.[l].dW
    let b = parameters.[l].b - α * gradients.[l].db
    acc |> Map.add l { W = W; b = b }

  arch.Layers
  |> List.mapi (fun i _ -> i + 1)
  |> List.fold _folder Map.empty

let trainNetwork (seed: int) (callback: EpochCallback) (arch: Architecture) (X: Matrix<double>) (Y: Matrix<double>) (hp: HyperParameters): Parameters =
  let timer = Stopwatch()
  let _folder parameters epoch =
    timer.Restart()
    let Ŷ, caches = _forwardPropagate arch parameters X
    let J = _computeCost Y Ŷ
    let gs = _backwardPropagate arch Y Ŷ caches
    let parameters = _updateParameters arch hp.α parameters gs
    timer.Stop()
    let accuracy = _computeAccuracy Y Ŷ
    callback epoch timer.Elapsed J accuracy
    parameters


  let ps0 = arch |> _initializeParameters seed

  seq { for epoch in 0 .. (hp.Epochs - 1) do epoch }
  |> Seq.fold _folder ps0

(*
TODO
- Refactor out relu/sigmoid and their backwards
- unit tests are independent of activation functions
- compare perf with numpy
- gradient checking
*)
