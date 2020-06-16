module YannLib.DNN

open System
open System.Diagnostics
open MathNet.Numerics.Distributions
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics

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

let εDivBy0Gaurd = 1E-12

let bugCheck<'T> : 'T = raise (InvalidOperationException("This case cannot occur. It is a bug in the code."))

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

type BatchSize =
  | BatchSize1
  | BatchSize64
  | BatchSize128
  | BatchSize256
  | BatchSize512
  | BatchSize1024
  | BatchSizeAll
  with
  member this.toInt max =
    match this with
    | BatchSize1 -> 1
    | BatchSize64 -> 64
    | BatchSize128 -> 128
    | BatchSize256 -> 256
    | BatchSize512 -> 512
    | BatchSize1024 -> 1024
    | BatchSizeAll -> max

type MomentumParameters = 
  { β: double }
  static member Defaults: MomentumParameters =
    { β = 0.9 }

type ADAMParameters = 
  { β1: double; β2: double; ε: double }
  static member Defaults: ADAMParameters =
    { β1 = 0.9; β2 = 0.999; ε = 1e-8 }

type Optimization =
  | NoOptimization
  | MomentumOptimization of MomentumParameters
  | ADAMOptimization of ADAMParameters

type HyperParameters =
  { Epochs : int
    α: double
    HeScale: double
    λ: double option
    Optimization: Optimization
    BatchSize: BatchSize } with
  static member Defaults = 
    { Epochs = 1_000
      α = 0.01
      HeScale = 1.
      λ = Some 0.7
      Optimization = ADAMOptimization ADAMParameters.Defaults
      BatchSize = BatchSize64 } 

[<DebuggerDisplay("W = {W.ShapeString()}, b = {b.ShapeString()}")>]
type Parameter = { W: Matrix<double>; b: Vector<double> }
type Parameters = Map<int, Parameter>

/// Exponentially weighted average of the gradients.
type GradientVelocity = { dWv: Matrix<double>; dbv: Vector<double> }
type GradientVelocities = Map<int, GradientVelocity>

/// Exponentially weighted average of the squared gradient.
type SquaredGradientVelocity = { dWs: Matrix<double>; dbs: Vector<double> }
type SquaredGradientVelocities = Map<int, SquaredGradientVelocity>

type TrainingState =
  | NoOptTrainingState of Parameters
  | MomentumTrainingState of Parameters * GradientVelocities
  | ADAMTrainingState of Parameters * GradientVelocities * SquaredGradientVelocities * double
  with
  member this.Parameters =
    match this with
    | NoOptTrainingState p -> p
    | MomentumTrainingState (p, _) -> p
    | ADAMTrainingState (p, _, _, _) -> p

[<DebuggerDisplay("Aprev = {Aprev.ShapeString()}, W = {W.ShapeString()}, b = {b.ShapeString()}, Z = {Z.ShapeString()}")>]
type Cache = { Aprev: Matrix<double>; D: Matrix<double> option; W: Matrix<double>; b: Vector<double>; Z: Matrix<double> }
type Caches = Map<int, Cache>
let _invalidCache = { Aprev = _invalidMatrix; D = None; W = _invalidMatrix; b = _invalidVector; Z = _invalidMatrix }

[<DebuggerDisplay("dA = {dA.ShapeString()}, dW = {dW.ShapeString()}, db = {db.ShapeString()}")>]
type Gradient = { dA: Matrix<double>; dW: Matrix<double>; db: Vector<double> }
type Gradients = Map<int, Gradient>
let _invalidGradient = { dA = _invalidMatrix; dW = _invalidMatrix; db = _invalidVector }

type EpochCallback = int -> TimeSpan -> double -> double -> unit

let _initializeParameters heScale (seed: int) (arch: Architecture): Parameters =
  let ws =
    seq { yield arch.nₓ; yield! arch.Layers |> Seq.map (fun l -> l.n) }
    |> Seq.pairwise
    |> Seq.mapi (fun i (nPrev, n) -> Matrix<double>.Build.Random(n, nPrev, Normal.WithMeanVariance(0.0, 1.0, Random(seed + i))) * Math.Sqrt(2.0 / double nPrev) * heScale)

  let bs =
    arch.Layers
    |> Seq.map (fun l -> Vector<double>.Build.Dense(l.n, 0.0))

  Seq.zip ws bs
  |> Seq.mapi (fun i (W, b) -> i + 1, { W = W; b = b })
  |> Map.ofSeq

let _initializeGradientVelocities (arch: Architecture): GradientVelocities =
  let ws =
    seq { yield arch.nₓ; yield! arch.Layers |> Seq.map (fun l -> l.n) }
    |> Seq.pairwise
    |> Seq.map (fun (nPrev, n) -> Matrix<double>.Build.Dense(n, nPrev, 0.))

  let bs =
    arch.Layers
    |> Seq.map (fun l -> Vector<double>.Build.Dense(l.n, 0.))

  Seq.zip ws bs
  |> Seq.mapi (fun i (dW, db) -> i + 1, { dWv = dW; dbv = db })
  |> Map.ofSeq

let _initializeSquaredGradientVelocities (arch: Architecture): SquaredGradientVelocities =
  let wss =
    seq { yield arch.nₓ; yield! arch.Layers |> Seq.map (fun l -> l.n) }
    |> Seq.pairwise
    |> Seq.map (fun (nPrev, n) -> Matrix<double>.Build.Dense(n, nPrev, 0.))

  let bss =
    arch.Layers
    |> Seq.map (fun l -> Vector<double>.Build.Dense(l.n, 0.))

  Seq.zip wss bss
  |> Seq.mapi (fun i (dW, db) -> i + 1, { dWs = dW; dbs = db })
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

let _linearActivationForward getKeepProb (Aprev: Matrix<double>) (W: Matrix<double>) (b: Vector<double>) (l, layer: Layer) =
  let Z = _linearForward Aprev W b
  let A = __activationsForward layer.Activation Z

  let kp = layer.KeepProb |> Option.bind getKeepProb
  let A, D = _adjustAForDropout A l kp

  A, { Aprev = Aprev; D = D; W = W; b = b; Z = Z }

let private __forwardPropagate getKeepProb arch (parameters: Parameters) (X: Matrix<double>): (Matrix<double> * Caches) =
  let _folder (Aprev: Matrix<double>, acc: Map<int, Cache>) (l, layer: Layer) =
    let (A, cache) = _linearActivationForward getKeepProb Aprev parameters.[l].W parameters.[l].b (l, layer)
    (A, acc |> Map.add l cache)

  seq { for l = 1 to arch.Layers.Length do yield l, arch.Layers.[l - 1] }
  |> Seq.fold _folder (X, Map.empty)

let _forwardPropagatePredict = __forwardPropagate (fun _ -> None)

let _forwardPropagateTrain = __forwardPropagate Some

let private __sumOfSquaresOfW (parameters: Parameters) m λ =
  λ / (2. * m) * (parameters |> Seq.fold (fun acc e -> acc + e.Value.W.PointwisePower(2.).ColumnSums().Sum()) 0.)

let _computeCost λ (Y: Matrix<double>) (Ŷ: Matrix<double>) (parameters: Parameters): double =
  let cost' =
    Y.TransposeAndMultiply(Ŷ.PointwiseLog()) +
    (1. - Y).TransposeAndMultiply((1. - Ŷ).PointwiseLog())
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
  | _ -> bugCheck

let _linearBackward λ (dZ: Matrix<double>) { Aprev = Aprev; W = W } =
  let m = Aprev.Shape().[1] |> double

  let dW = (1.0 / m) * dZ.TransposeAndMultiply(Aprev)
  let dW =
    match λ with
    | Some λ -> dW + ((λ / m) * W)
    | None -> dW
  let db = (1.0 / m) * (dZ.EnumerateColumns() |> Seq.reduce (+))
  let dAprev = W.TransposeThisAndMultiply(dZ)

  dAprev, dW, db

let _linearActivationBackward λ (dA: Matrix<double>) cache activation =
  let dZ = __activationsBackward activation cache.Z dA
  _linearBackward λ dZ cache

let _backwardPropagate arch λ (Y: Matrix<double>) (Ŷ: Matrix<double>) (caches: Caches): Gradients =
  let _folder (acc: Gradients) (l, layer: Layer) =
    let (dAprev, dW, db) = _linearActivationBackward λ acc.[l].dA caches.[l] layer.Activation
    let dAprev =
      if l > 1 then
        _adjustdAForDropout dAprev (caches.[l - 1].D, arch.Layers.[l - 2].KeepProb)
      else
        dAprev
    let gradPrev = { _invalidGradient with dA = dAprev }
    let grad = { acc.[l] with dW = dW; db = db }
    acc |> Map.remove l |> Map.add l grad |> Map.add (l - 1) gradPrev

  let L = arch.Layers.Length
  let dAL = - (Y.PointwiseDivide(Ŷ.Add(εDivBy0Gaurd)) - Y.Negate().Add(1.0).PointwiseDivide(Ŷ.Negate().Add(1.0 + εDivBy0Gaurd)))
  let dAL = _adjustdAForDropout dAL (caches.[L].D, arch.Layers.[L - 1].KeepProb)
  let g0 = Map.empty |> Map.add L { _invalidGradient with dA = dAL }

  seq { for l = arch.Layers.Length downto 1 do yield l, arch.Layers.[l - 1] }
  |> Seq.fold _folder g0

let _updateParametersWithNoOptimization arch (α: double) (parameters: Parameters) (gradients: Gradients): TrainingState =
  let _folder acc l =
    let W = parameters.[l].W - α * gradients.[l].dW
    let b = parameters.[l].b - α * gradients.[l].db
    acc |> Map.add l { W = W; b = b }

  seq { for l = 1 to arch.Layers.Length do yield l }
  |> Seq.fold _folder Map.empty
  |> NoOptTrainingState

let _updateParametersWithMomentum arch (α: double) (mp: MomentumParameters) (gradients: Gradients) (parameters: Parameters, v: GradientVelocities): TrainingState =
  let _folder (accp, accv) l =
    let dWv = mp.β * v.[l].dWv + (1. - mp.β) * gradients.[l].dW
    let dbv = mp.β * v.[l].dbv + (1. - mp.β) * gradients.[l].db

    let W = parameters.[l].W - α * dWv
    let b = parameters.[l].b - α * dbv
    (accp |> Map.add l { W = W; b = b }), (accv |> Map.add l { dWv = dWv; dbv = dbv })

  seq { for l = 1 to arch.Layers.Length do yield l }
  |> Seq.fold _folder (Map.empty, Map.empty)
  |> MomentumTrainingState

let _updateParametersWithADAM arch (α: double) (ap: ADAMParameters) (gradients: Gradients) (parameters: Parameters, v: GradientVelocities, s: SquaredGradientVelocities, t: double): TrainingState =
  let _folder (accp, accv, accs) l =
    let dWv = ap.β1 * v.[l].dWv + (1. - ap.β1) * gradients.[l].dW
    let dbv = ap.β1 * v.[l].dbv + (1. - ap.β1) * gradients.[l].db
    let dWv_corrected = dWv / (1. - ap.β1 ** t)
    let dbv_corrected = dbv / (1. - ap.β1 ** t)

    let dWs = ap.β2 * s.[l].dWs + (1. - ap.β2) * gradients.[l].dW.PointwisePower(2.)
    let dbs = ap.β2 * s.[l].dbs + (1. - ap.β2) * gradients.[l].db.PointwisePower(2.)
    let dWs_corrected = dWs / (1. - ap.β2 ** t)
    let dbs_corrected = dbs / (1. - ap.β2 ** t)

    let W = parameters.[l].W - α * dWv_corrected.PointwiseDivide(dWs_corrected.PointwisePower(0.5) + ap.ε)
    let b = parameters.[l].b - α * dbv_corrected.PointwiseDivide(dbs_corrected.PointwisePower(0.5) + ap.ε)
    (accp |> Map.add l { W = W; b = b }), (accv |> Map.add l { dWv = dWv; dbv = dbv }), (accs |> Map.add l { dWs = dWs; dbs = dbs })

  let p, v, s = 
    seq { for l = 1 to arch.Layers.Length do yield l }
    |> Seq.fold _folder (Map.empty, Map.empty, Map.empty)

  (p, v, s, t + 1.) |> ADAMTrainingState

let _updateParameters arch (hp: HyperParameters) (ts: TrainingState) (gradients: Gradients): TrainingState =
  match hp.Optimization with
  | NoOptimization -> _updateParametersWithNoOptimization arch hp.α ts.Parameters gradients
  | MomentumOptimization mp -> _updateParametersWithMomentum arch hp.α mp gradients (match ts with | MomentumTrainingState (p, v) -> (p, v) | _ -> bugCheck)
  | ADAMOptimization ap -> _updateParametersWithADAM arch hp.α ap gradients (match ts with | ADAMTrainingState (p, v, s, t) -> (p, v, s, t) | _ -> bugCheck)

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
    let Ŷ, _ = _forwardPropagateTrain arch p X
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

let predict arch X parameters = 
  let Ŷ, _ = _forwardPropagatePredict arch parameters X
  Ŷ.PointwiseRound()

let computeAccuracy arch (X: Matrix<double>) (Y: Matrix<double>) parameters =
  let Ŷ = predict arch X parameters
  1. - Y.Subtract(Ŷ).ColumnAbsoluteSums().Sum() / (Y.RowCount * Y.ColumnCount |> double)

let _getMiniBatches (batchSize: BatchSize) (X: Matrix<double>, Y: Matrix<double>): (Matrix<double> * Matrix<double>) seq =
  let bs = batchSize.toInt X.ColumnCount
  seq {
    let nComplete = X.ColumnCount / bs
    for bn = 0 to nComplete - 1 do
      let c0 = bn * bs
      let c1 = c0 + bs - 1
      yield X.[*, c0 .. c1], Y.[*, c0 .. c1]

    if X.ColumnCount % bs <> 0 then
      let c0 = nComplete * bs
      let c1 = X.ColumnCount - 1
      yield X.[*, c0 .. c1], Y.[*, c0 .. c1]
  }

let shuffleDataSet_ seed (X: Matrix<double>, Y: Matrix<double>)  =
  let permutation = Permutation(Combinatorics.GeneratePermutation(X.ColumnCount, Random(seed)))
  X.PermuteColumns(permutation)
  Y.PermuteColumns(permutation)

let private __trainNetworkFor1MiniBatch arch hp (J: double, ts: TrainingState, timer: Stopwatch) (X: Matrix<double>, Y: Matrix<double>): (double * TrainingState * Stopwatch) =
  timer.Start()
  let Ŷ, caches = _forwardPropagateTrain arch ts.Parameters X
  let gradients = _backwardPropagate arch hp.λ Y Ŷ caches
  let ts = _updateParameters arch hp ts gradients
  timer.Stop()
  let m = X.ColumnCount
  let J = J + (double m) * (_computeCost hp.λ Y Ŷ ts.Parameters)
  J, ts, timer

let private __trainNetworkFor1Epoch seed (timer: Stopwatch) (callback: EpochCallback) (arch: Architecture) (X: Matrix<double>) (Y: Matrix<double>) (hp: HyperParameters) (ts: TrainingState) epoch =
  let m = X.ColumnCount
  timer.Restart()
  // NOTE: Should we move it out of there?
  // TODO: Both deeplearning.ai and neuralnetworksanddeeplearning.com do it on every epoc. Ask on various forums what is the best practice if dataset is large?
  let X = X.Clone()
  let Y = Y.Clone()
  shuffleDataSet_ (seed + epoch) (X, Y)

  let J, ts, timer = 
    (X, Y)
    |> _getMiniBatches hp.BatchSize
    |> Seq.fold (__trainNetworkFor1MiniBatch arch hp) (0., ts, timer)
  timer.Stop()
  let J = J / double m
  let accuracy = computeAccuracy arch X Y ts.Parameters
  callback epoch timer.Elapsed J accuracy
  ts

let trainNetwork seed p0 (callback: EpochCallback) (arch: Architecture) (hp: HyperParameters) (X: Matrix<double>) (Y: Matrix<double>): Parameters =
  assert (X.ColumnCount = Y.ColumnCount)

  let timer = Stopwatch()

  let p0 = p0 |> Option.defaultValue (_initializeParameters hp.HeScale seed arch)

  let ts0 =
    match hp.Optimization with
    | NoOptimization -> NoOptTrainingState p0
    | MomentumOptimization -> MomentumTrainingState (p0, _initializeGradientVelocities arch)
    | ADAMOptimization -> ADAMTrainingState (p0, _initializeGradientVelocities arch, _initializeSquaredGradientVelocities arch, 1.)

  let ts =
    seq { for epoch in 0 .. (hp.Epochs - 1) do epoch }
    |> Seq.fold (__trainNetworkFor1Epoch seed timer callback arch X Y hp) ts0

  ts.Parameters

let calculateDeltaForGradientCheck λ ε arch (parameters: Parameters) (X: Matrix<double>) (Y: Matrix<double>) =
  let grads' =
    _calculateNumericalGradients λ ε arch parameters X Y
    |> Seq.sortBy (fun kv -> kv.Key)
    |> Seq.map (fun kv -> kv.Value)
    |> Vector<double>.Build.DenseOfEnumerable

  let Ŷ, caches = _forwardPropagateTrain arch parameters X
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
