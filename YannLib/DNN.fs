module YannLib.DNN

open System
open System.Diagnostics
open MathNet.Numerics.Distributions
open MathNet.Numerics.LinearAlgebra

(*

  # Notation & Dimensional Analysis

  ## Formula

  m  = # of training input feature vectors
  nₓ = Dimension of input feature vector
  X = Input
  A₀ = X
  Zₗ = Wₗ.Aₗ₋₁ + bₗ
  Aₗ = gₗ(Zₗ)
  Ŷ = Output of the last layer
  Y = Expected output
  L = # layers

  dim(X) = nₓ x m
  dim(Wₗ) = nₗ x nₗ₋₁
  dim(bₗ) = nₗ x 1
  dim(Zₗ) = nₗ₋₁ x m
  dim(Aₗ) = dim(Zₗ)

  ## Example

  nₓ      = 3
  Layers  = 4, 5, 2
  m       = 7
  L       = 3

  [  W1    A0    b1      A1   ]    [  W2    A1    b2      A2   ]    [  W3    A2    b3   =  A3   ]
  [ (4x3).(3x7)+(4x1) = (4x7) ] -> [ (5x4).(4x7)+(5x1) = (5x7) ] -> [ (2x5).(5x7)+(2x1) = (2x7) ]

 *)

let _invalidMatrix = Matrix<double>.Build.Sparse(1, 1, 0.0)
let _invalidVector = Vector<double>.Build.Sparse(1, 0.0)

type Activation =
  | ReLU
  | Sigmoid

let _activationsForward : Map<Activation, Matrix<double> -> Matrix<double>> =
  [ (ReLU, Activations.ReLU.forward)
    (Sigmoid, Activations.Sigmoid.forward) ] |> Map.ofList

let _activationsBackward : Map<Activation, Matrix<double> -> Matrix<double> -> Matrix<double>> =
  [ (ReLU, Activations.ReLU.backward)
    (Sigmoid, Activations.Sigmoid.backward) ] |> Map.ofList

type Layer =
  { n: int
    Activation: Activation }

type Architecture =
  { nₓ: int
    Layers: Layer list }

type HyperParameters =
  { Epochs : int
    α: double
    λ: double }

[<DebuggerDisplay("W = {W.ShapeString()}, b = {b.ShapeString()}")>]
type Parameter = { W: Matrix<double>; b: Vector<double> }
type Parameters = Map<int, Parameter>

type ParameterInitialization = 
  | Parameters of Parameters
  | Seed of int

[<DebuggerDisplay("Aprev = {Aprev.ShapeString()}, W = {W.ShapeString()}, b = {b.ShapeString()}, Z = {Z.ShapeString()}")>]
type Cache = { Aprev: Matrix<double>; W: Matrix<double>; b: Vector<double>; Z: Matrix<double> }
type Caches = Map<int, Cache>
let _invalidCache = { Aprev = _invalidMatrix; W = _invalidMatrix; b = _invalidVector; Z = _invalidMatrix }

[<DebuggerDisplay("dA = {dA.ShapeString()}, dW = {dW.ShapeString()}, db = {db.ShapeString()}")>]
type Gradient = { dA: Matrix<double>; dW: Matrix<double>; db: Vector<double> }
type Gradients = Map<int, Gradient>
let _invalidGradient = { dA = _invalidMatrix; dW = _invalidMatrix; db = _invalidVector }

type EpochCallback = int -> TimeSpan -> double -> double -> unit

let _initializeParameters (seed: int) (arch: Architecture): Parameters =
  let ws =
    seq { yield arch.nₓ; yield! arch.Layers |> Seq.map (fun l -> l.n) }
    |> Seq.pairwise
    |> Seq.mapi (fun i (nPrev, n) -> Matrix<double>.Build.Random(n, nPrev, Normal.WithMeanVariance(0.0, 1.0, Random(seed + i))) * Math.Sqrt(2.0 / double nPrev))

  let bs =
    arch.Layers
    |> Seq.map (fun l -> Vector<double>.Build.Dense(l.n, 0.0))

  Seq.zip ws bs
  |> Seq.mapi (fun i (W, b) -> i + 1, { W = W; b = b })
  |> Map.ofSeq

let _linearForward (Aprev: Matrix<double>) (W: Matrix<double>) (b: Vector<double>) =
  let Z = W.Multiply(Aprev) + b.BroadcastC(Aprev.ColumnCount)
  { Aprev = Aprev; W = W; b = b; Z = Z }

let _linearActivationForward (Aprev: Matrix<double>) (W: Matrix<double>) (b: Vector<double>) activation =
  let cache = _linearForward Aprev W b
  let A = _activationsForward.[activation] cache.Z
  A, cache

let _forwardPropagate arch (parameters: Parameters) (X: Matrix<double>): (Matrix<double> * Caches) =
  let _folder (APrev: Matrix<double>, acc: Map<int, Cache>) (l, layer: Layer) =
    let (A, cache) = _linearActivationForward APrev parameters.[l].W parameters.[l].b layer.Activation
    (A, acc |> Map.add l cache)

  arch.Layers
  |> List.mapi (fun i l -> (i + 1), l)
  |> List.fold _folder (X, Map.empty)

let _computeCost (λ: double) (Y: Matrix<double>) (Ŷ: Matrix<double>) (parameters: Parameters): double =
  let cost' =
    Y.Multiply(Ŷ.PointwiseLog().Transpose()) +
    Y.Negate().Add(1.0).Multiply(Ŷ.Negate().Add(1.0).PointwiseLog().Transpose())
  assert ((cost'.RowCount, cost'.ColumnCount) = (1, 1))

  let m = double Y.ColumnCount
  let cost = (-1. / m) * cost'.Item(0, 0)
  let l2RegCost = λ / (2. * m) * (parameters |> Seq.fold (fun acc e -> acc + e.Value.W.PointwisePower(2.).ColumnSums().Sum()) 0.)

  cost + l2RegCost

let _computeAccuracy (Y: Matrix<double>) (Ŷ: Matrix<double>) =
  0.0

let _linearBackward (λ: double) (dZ: Matrix<double>) { Aprev = Aprev; W = W } =
  let m = Aprev.Shape().[1] |> double

  let dW = (1.0 / m) * dZ.Multiply(Aprev.Transpose()) + ((λ / m) * W)
  let db = (1.0 / m) * (dZ.EnumerateColumns() |> Seq.reduce (+))
  let dAprev = W.Transpose().Multiply(dZ)
  
  dAprev, dW, db

let _linearActivationBackward λ (dA: Matrix<double>) cache activation =
  let dZ = _activationsBackward.[activation] cache.Z dA
  _linearBackward λ dZ cache

let _backwardPropagate arch λ (Y: Matrix<double>) (Ŷ: Matrix<double>) (caches: Caches): Gradients =
  let _folder (acc: Gradients) (l, layer: Layer) =
    let (dAprev, dW, db) = _linearActivationBackward λ acc.[l].dA caches.[l] layer.Activation
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

// NOTE: Inspiration from deeplearning.ai & https://stats.stackexchange.com/questions/332089/numerical-gradient-checking-best-practices
let _calculateNumericalGradients λ ε arch (parameters: Parameters) (X: Matrix<double>) (Y: Matrix<double>) =
  let updateParamsW (l, r, c) ε =
    // NOTE: This clone will happen r x c times per W. Is there a better way?
    let W' = parameters.[l].W.Clone()
    W'.[r, c] <- W'.[r, c] + ε
    parameters |> Map.remove l |> Map.add l { parameters.[l] with W = W'  }

  let updateParamsb (l, i, _) ε =
    // NOTE: This clone will happen r x c times per b. Is there a better way?
    let b' = parameters.[l].b.Clone()
    b'.[i] <- b'.[i] + ε
    parameters |> Map.remove l |> Map.add l { parameters.[l] with b = b'  }

  let getCost updateParams index ε =
    let p = updateParams index ε
    let Ŷ, _ = _forwardPropagate arch p X
    _computeCost λ Y Ŷ parameters

  let _folder ptype updateParams acc index =
    let Jpos = getCost updateParams index ε
    let Jneg = getCost updateParams index -ε
    let grad' = (Jpos - Jneg) / (2.0 * ε)
    acc |> Map.add (ptype, index) grad'

  let gradsW =
    seq {
      for kv in parameters do
        for r in 0 .. (kv.Value.W.RowCount - 1) do
          for c in 0 .. (kv.Value.W.ColumnCount - 1) do
            yield kv.Key, r, c
    }
    |> Seq.fold (_folder 'W' updateParamsW) Map.empty

  let gradsWb = 
    seq {
      for kv in parameters do
        for i in 0 .. (kv.Value.b.Count - 1) do
          yield kv.Key, i, 0
    }
    |> Seq.fold (_folder 'b' updateParamsb) gradsW

  gradsWb

let trainNetwork paramsInit (callback: EpochCallback) (arch: Architecture) (X: Matrix<double>) (Y: Matrix<double>) (hp: HyperParameters): Parameters =
  let timer = Stopwatch()
  let _folder parameters epoch =
    timer.Restart()
    let Ŷ, caches = _forwardPropagate arch parameters X
    let gradients = _backwardPropagate arch hp.λ Y Ŷ caches
    let parameters = _updateParameters arch hp.α parameters gradients
    timer.Stop()
    let J = _computeCost hp.λ Y Ŷ parameters
    callback epoch timer.Elapsed J Double.NaN
    parameters

  let ps0 = 
    match paramsInit with
    | Parameters ps -> ps
    | Seed s -> arch |> _initializeParameters s

  seq { for epoch in 0 .. (hp.Epochs - 1) do epoch }
  |> Seq.fold _folder ps0

let predit arch X parameters = 
  let Ŷ, _ = _forwardPropagate arch parameters X

  Ŷ.PointwiseRound()

let computeAccuracy arch (X: Matrix<double>) (Y: Matrix<double>) parameters =
  let Ŷ = predit arch X parameters

  Ŷ.Subtract(Y).ColumnAbsoluteSums().Sum() / (Y.RowCount * Ŷ.ColumnCount |> double)

let calculateDeltaForGradientCheck λ ε arch (parameters: Parameters) (X: Matrix<double>) (Y: Matrix<double>) =
  let grads' =
    _calculateNumericalGradients λ ε arch parameters X Y
    |> Seq.sortBy (fun kv -> kv.Key)
    |> Seq.map (fun kv -> kv.Value)
    |> Vector<double>.Build.DenseOfEnumerable

  let Ŷ, caches = _forwardPropagate arch parameters X
  let gradients = _backwardPropagate arch λ Y Ŷ caches |> Map.remove 0

  let gradsW =
    seq {
      for kv in gradients do
        for r in 0 .. (kv.Value.dW.RowCount - 1) do
          for c in 0 .. (kv.Value.dW.ColumnCount - 1) do
            yield kv.Value.dW.[r, c]
    }
  let gradsb =
    seq {
      for kv in gradients do
        for i in 0 .. (kv.Value.db.Count - 1) do
          yield kv.Value.db.[i]
    }
  let grads = 
    Seq.append gradsW gradsb
    |> Vector<double>.Build.DenseOfEnumerable

  (grads - grads').L2Norm() / (grads.L2Norm() + grads'.L2Norm())
