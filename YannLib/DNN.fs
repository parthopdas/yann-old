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

type Layer =
  { n: int
    Activation: Activation
    KeepProb: double option }

type Architecture =
  { nₓ: int
    Layers: Layer array }

type HyperParameters =
  { Epochs : int
    α: double
    λ: double option }

[<DebuggerDisplay("W = {W.ShapeString()}, b = {b.ShapeString()}")>]
type Parameter = { W: Matrix<double>; b: Vector<double> }
type Parameters = Map<int, Parameter>

type ParameterInitialization = 
  | Parameters of Parameters
  | Seed of int

[<DebuggerDisplay("Aprev = {Aprev.ShapeString()}, W = {W.ShapeString()}, b = {b.ShapeString()}, Z = {Z.ShapeString()}")>]
type Cache = { Aprev: Matrix<double>; D: Matrix<double> option; W: Matrix<double>; b: Vector<double>; Z: Matrix<double> }
type Caches = Map<int, Cache>
let _invalidCache = { Aprev = _invalidMatrix; D = None; W = _invalidMatrix; b = _invalidVector; Z = _invalidMatrix }

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

let private __activationsForward = function
  | ReLU -> Activations.ReLU.forward
  | Sigmoid -> Activations.Sigmoid.forward

let private __activationsBackward = function
  | ReLU -> Activations.ReLU.backward
  | Sigmoid -> Activations.Sigmoid.backward

let _linearForward (Aprev: Matrix<double>) (W: Matrix<double>) (b: Vector<double>) =
  let Z = W.Multiply(Aprev) + b.BroadcastC(Aprev.ColumnCount)
  Z

let _adjustAForDropout (A: Matrix<double>) l = function
  | Some kp ->
    let D = Matrix<double>.Build.Random(A.RowCount, A.ColumnCount, ContinuousUniform(0.0, 1.0, Random(l * l)))
    D.MapInplace(fun v -> if v < kp then 1. else 0.)

    let A = A.PointwiseMultiply(D).Divide(kp)
    A, Some D
  | None -> A, None

let _linearActivationForward pMode (Aprev: Matrix<double>) (W: Matrix<double>) (b: Vector<double>) (l, layer: Layer) =
  let Z = _linearForward Aprev W b
  let A = __activationsForward layer.Activation Z

  let kp = layer.KeepProb |> Option.bind (fun x -> if pMode then None else Some x)
  let A, D = _adjustAForDropout A l kp

  A, { Aprev = Aprev; D = D; W = W; b = b; Z = Z }

let _forwardPropagate pMode arch (parameters: Parameters) (X: Matrix<double>): (Matrix<double> * Caches) =
  let _folder (Aprev: Matrix<double>, acc: Map<int, Cache>) (l, layer: Layer) =
    let (A, cache) = _linearActivationForward pMode Aprev parameters.[l].W parameters.[l].b (l, layer)
    (A, acc |> Map.add l cache)

  arch.Layers
  |> Array.mapi (fun i l -> (i + 1), l)
  |> Array.fold _folder (X, Map.empty)

let private __sumOfSquaresOfW (parameters: Parameters) m λ =
  λ / (2. * m) * (parameters |> Seq.fold (fun acc e -> acc + e.Value.W.PointwisePower(2.).ColumnSums().Sum()) 0.)

let _computeCost λ (Y: Matrix<double>) (Ŷ: Matrix<double>) (parameters: Parameters): double =
  let cost' =
    Y.Multiply(Ŷ.PointwiseLog().Transpose()) +
    Y.Negate().Add(1.0).Multiply(Ŷ.Negate().Add(1.0).PointwiseLog().Transpose())
  assert ((cost'.RowCount, cost'.ColumnCount) = (1, 1))

  let m = double Y.ColumnCount

  let cost = (-1. / m) * cost'.Item(0, 0)
  let l2RegCost =
    match λ with
    | Some λ -> __sumOfSquaresOfW parameters m λ
    | None -> 0.

  cost + l2RegCost

let _adjustdAForDropout (dA: Matrix<double>) = function
  | Some D, Some kp -> dA.PointwiseMultiply(D).Divide(kp)
  | None, None -> dA
  | _ -> failwithf "kp and D both need to be None or Some"

let _linearBackward λ (dZ: Matrix<double>) { Aprev = Aprev; W = W } =
  let m = Aprev.Shape().[1] |> double

  let dW = (1.0 / m) * dZ.Multiply(Aprev.Transpose())
  let dW =
    match λ with
    | Some λ -> dW + ((λ / m) * W)
    | None -> dW
  let db = (1.0 / m) * (dZ.EnumerateColumns() |> Seq.reduce (+))
  let dAprev = W.Transpose().Multiply(dZ)

  dAprev, dW, db

let _linearActivationBackward λ (dA: Matrix<double>) cache activation =
  let dZ = __activationsBackward activation cache.Z dA
  _linearBackward λ dZ cache

let _backwardPropagate arch λ (Y: Matrix<double>) (Ŷ: Matrix<double>) (caches: Caches): Gradients =
  let _folder (acc: Gradients) (l, layer: Layer) =
    let (dAprev, dW, db) = _linearActivationBackward λ acc.[l].dA caches.[l] layer.Activation
    let dAprev =
      if l >= 2 then
        _adjustdAForDropout dAprev (caches.[l - 1].D, arch.Layers.[l - 2].KeepProb)
      else
        dAprev
    let gradPrev = { _invalidGradient with dA = dAprev }
    let grad = { acc.[l] with dW = dW; db = db }
    acc |> Map.remove l |> Map.add l grad |> Map.add (l - 1) gradPrev

  let L = arch.Layers.Length
  let dAL = - (Y.PointwiseDivide(Ŷ) - Y.Negate().Add(1.0).PointwiseDivide(Ŷ.Negate().Add(1.0)))
  let dAL = _adjustdAForDropout dAL (caches.[L].D, arch.Layers.[L - 1].KeepProb)
  let g0 = Map.empty |> Map.add L { _invalidGradient with dA = dAL }

  seq { for i = arch.Layers.Length downto 1 do yield i, arch.Layers.[i - 1] }
  |> Seq.fold _folder g0

let _updateParameters arch (α: double) (parameters: Parameters) (gradients: Gradients): Parameters =
  let _folder acc l =
    let W = parameters.[l].W - α * gradients.[l].dW
    let b = parameters.[l].b - α * gradients.[l].db
    acc |> Map.add l { W = W; b = b }

  arch.Layers
  |> Array.mapi (fun i _ -> i + 1)
  |> Array.fold _folder Map.empty

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
    let Ŷ, _ = _forwardPropagate false arch p X
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

let initializeHyperParameters () =
  { Epochs = 0
    α = 0.01
    λ = Some 0.7 }

let predict arch X parameters = 
  let Ŷ, _ = _forwardPropagate true arch parameters X
  Ŷ.PointwiseRound()

let computeAccuracy arch (X: Matrix<double>) (Y: Matrix<double>) parameters =
  let Ŷ = predict arch X parameters
  1. - Y.Subtract(Ŷ).ColumnAbsoluteSums().Sum() / (Y.RowCount * Y.ColumnCount |> double)

let trainNetwork paramsInit (callback: EpochCallback) (arch: Architecture) (X: Matrix<double>) (Y: Matrix<double>) (hp: HyperParameters): Parameters =
  let timer = Stopwatch()
  let _folder parameters epoch =
    timer.Restart()
    let Ŷ, caches = _forwardPropagate false arch parameters X
    let gradients = _backwardPropagate arch hp.λ Y Ŷ caches
    let parameters = _updateParameters arch hp.α parameters gradients
    timer.Stop()
    let J = _computeCost hp.λ Y Ŷ parameters
    let accuracy = computeAccuracy arch X Y parameters
    callback epoch timer.Elapsed J accuracy
    parameters

  let ps0 = 
    match paramsInit with
    | Parameters ps -> ps
    | Seed s -> arch |> _initializeParameters s

  seq { for epoch in 0 .. (hp.Epochs - 1) do epoch }
  |> Seq.fold _folder ps0

let calculateDeltaForGradientCheck λ ε arch (parameters: Parameters) (X: Matrix<double>) (Y: Matrix<double>) =
  let grads' =
    _calculateNumericalGradients λ ε arch parameters X Y
    |> Seq.sortBy (fun kv -> kv.Key)
    |> Seq.map (fun kv -> kv.Value)
    |> Vector<double>.Build.DenseOfEnumerable

  let Ŷ, caches = _forwardPropagate false arch parameters X
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
