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

type HyperParameters =
  { Epochs : int
    α: double
    HeScale: double
    λ: double option
    BatchSize: BatchSize } with
  static member CreateWithDefaults () = 
    { Epochs = 1_000
      α = 0.01
      HeScale = 1.
      λ = Some 0.7
      BatchSize = BatchSize128 }

[<DebuggerDisplay("W = {W.ShapeString()}, b = {b.ShapeString()}")>]
type Parameter = { W: Matrix<double>; b: Vector<double> }
type Parameters = Map<int, Parameter>

type Vs = Map<int, Parameter>

type Ss = Map<int, Parameter>

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

let _initializeVs (arch: Architecture): Vs =
  let ws =
    seq { yield arch.nₓ; yield! arch.Layers |> Seq.map (fun l -> l.n) }
    |> Seq.pairwise
    |> Seq.map (fun (nPrev, n) -> Matrix<double>.Build.Dense(n, nPrev, 0.))

  let bs =
    arch.Layers
    |> Seq.map (fun l -> Vector<double>.Build.Dense(l.n, 0.))

  Seq.zip ws bs
  |> Seq.mapi (fun i (W, b) -> i + 1, { W = W; b = b })
  |> Map.ofSeq

let _initializeSs (arch: Architecture): Vs * Ss =
  let wsv =
    seq { yield arch.nₓ; yield! arch.Layers |> Seq.map (fun l -> l.n) }
    |> Seq.pairwise
    |> Seq.map (fun (nPrev, n) -> Matrix<double>.Build.Dense(n, nPrev, 0.))

  let bsv =
    arch.Layers
    |> Seq.map (fun l -> Vector<double>.Build.Dense(l.n, 0.))

  let v =
    Seq.zip wsv bsv
    |> Seq.mapi (fun i (W, b) -> i + 1, { W = W; b = b })
    |> Map.ofSeq

  let wss =
    seq { yield arch.nₓ; yield! arch.Layers |> Seq.map (fun l -> l.n) }
    |> Seq.pairwise
    |> Seq.map (fun (nPrev, n) -> Matrix<double>.Build.Dense(n, nPrev, 0.))

  let bss =
    arch.Layers
    |> Seq.map (fun l -> Vector<double>.Build.Dense(l.n, 0.))

  let s = 
    Seq.zip wss bss
    |> Seq.mapi (fun i (W, b) -> i + 1, { W = W; b = b })
    |> Map.ofSeq

  v, s

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
  | _ -> failwithf "kp and D both need to be None or Some. This is a bug with the code."

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

let _updateParameters arch (α: double) (parameters: Parameters) (gradients: Gradients): Parameters =
  let _folder acc l =
    let W = parameters.[l].W - α * gradients.[l].dW
    let b = parameters.[l].b - α * gradients.[l].db
    acc |> Map.add l { W = W; b = b }

  seq { for l = 1 to arch.Layers.Length do yield l }
  |> Seq.fold _folder Map.empty

let _updateParametersMomentum arch (α: double) (β: double) (gradients: Gradients) (v: Vs) (parameters: Parameters) : Parameters * Vs =
  let _folder (accp, accv) l =
    let dWv = v.[l].W.Multiply(β) + gradients.[l].dW.Multiply(1. - β)
    let dbv = v.[l].b.Multiply(β) + gradients.[l].db.Multiply(1. - β)

    // TODO: Change W, b in Vs to dW and db
    let W = parameters.[l].W - α * dWv
    let b = parameters.[l].b - α * dbv
    (accp |> Map.add l { W = W; b = b }), (accv |> Map.add l { W = dWv; b = dbv })

  seq { for l = 1 to arch.Layers.Length do yield l }
  |> Seq.fold _folder (Map.empty, Map.empty)

// TODO: Change W, b in Vs to dW and db
// TODO: Change W, b in Ss to dW and db
let _updateParametersADAM arch (α: double) (β1: double) (β2: double) (ε: double) (t: double) (gradients: Gradients) (v: Vs) (s: Ss) (parameters: Parameters) : Parameters * Vs * Ss =
  let _folder (accp, accv, accs) l =
    let dWv = v.[l].W.Multiply(β1) + gradients.[l].dW.Multiply(1. - β1)
    let dbv = v.[l].b.Multiply(β1) + gradients.[l].db.Multiply(1. - β1)
    let dWv_corrected = dWv / (1. - β1 ** t)
    let dbv_corrected = dbv / (1. - β1 ** t)

    let dWs = β2 * s.[l].W + (1. - β2) * gradients.[l].dW.PointwisePower(2.)
    let dbs = β2 * s.[l].b + (1. - β2) * gradients.[l].db.PointwisePower(2.)
    let dWs_corrected = dWs / (1. - β2 ** t)
    let dbs_corrected = dbs / (1. - β2 ** t)

    let W = parameters.[l].W - α * dWv_corrected.PointwiseDivide(dWs_corrected.PointwisePower(0.5) + ε)
    let b = parameters.[l].b - α * dbv_corrected.PointwiseDivide(dbs_corrected.PointwisePower(0.5) + ε)
    (accp |> Map.add l { W = W; b = b }), (accv |> Map.add l { W = dWv; b = dbv }), (accs |> Map.add l { W = dWs; b = dbs })

  seq { for l = 1 to arch.Layers.Length do yield l }
  |> Seq.fold _folder (Map.empty, Map.empty, Map.empty)

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

let private __bsToInt max = function
  | BatchSize1 -> 1
  | BatchSize64 -> 64
  | BatchSize128 -> 128
  | BatchSize256 -> 256
  | BatchSize512 -> 512
  | BatchSize1024 -> 1024
  | BatchSizeAll -> max

let _getMiniBatches batchSize (X: Matrix<double>, Y: Matrix<double>): (Matrix<double> * Matrix<double>) seq =
  let bs = batchSize |> __bsToInt X.ColumnCount
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

let mutable i = 0

let shuffleDataSet_ seed (X: Matrix<double>, Y: Matrix<double>)  =
  let parray =
    [| [|174; 113; 86; 239; 12; 252; 261; 39; 200; 61; 122; 279; 126; 184; 151; 188; 130; 218; 70; 16; 283; 189; 2; 150; 274; 135; 11; 278; 41; 154; 294; 173; 53; 14; 60; 25; 155; 212; 199; 194; 88; 232; 159; 77; 267; 255; 103; 54; 202; 62; 216; 121; 265; 137; 44; 115; 157; 165; 292; 69; 217; 52; 210; 22; 257; 256; 3; 55; 134; 26; 101; 97; 272; 51; 36; 13; 183; 186; 102; 273; 262; 28; 254; 21; 9; 235; 95; 20; 158; 45; 224; 29; 24; 164; 146; 112; 116; 237; 175; 33; 243; 291; 63; 123; 226; 66; 49; 229; 238; 298; 139; 299; 0; 98; 128; 85; 108; 143; 204; 83; 182; 4; 296; 57; 220; 99; 59; 179; 271; 288; 290; 46; 198; 120; 30; 228; 56; 180; 246; 169; 282; 127; 230; 144; 42; 250; 177; 76; 205; 7; 73; 219; 221; 233; 152; 32; 15; 94; 107; 234; 147; 264; 35; 241; 87; 206; 259; 170; 213; 201; 195; 285; 240; 268; 64; 10; 197; 275; 260; 93; 281; 284; 5; 208; 193; 203; 58; 104; 225; 6; 276; 149; 163; 214; 74; 207; 190; 111; 293; 106; 148; 289; 244; 65; 185; 141; 156; 132; 100; 211; 31; 118; 119; 286; 167; 38; 222; 277; 82; 50; 84; 247; 251; 162; 75; 124; 105; 266; 129; 295; 196; 8; 40; 172; 249; 110; 47; 270; 48; 18; 43; 17; 258; 166; 90; 117; 153; 19; 78; 160; 114; 72; 68; 209; 215; 136; 131; 1; 187; 227; 178; 181; 133; 223; 248; 168; 263; 142; 245; 27; 138; 89; 23; 96; 79; 287; 192; 242; 171; 67; 145; 297; 81; 231; 37; 140; 34; 125; 236; 253; 176; 92; 109; 280; 71; 161; 269; 91; 80; 191|]
       [|224; 136; 67; 231; 213; 77; 1; 281; 191; 190; 75; 265; 250; 37; 257; 295; 90; 268; 21; 42; 169; 199; 78; 276; 48; 117; 255; 12; 194; 106; 186; 11; 193; 111; 166; 3; 220; 14; 293; 144; 86; 6; 16; 178; 236; 267; 229; 125; 7; 41; 210; 135; 272; 133; 34; 218; 35; 52; 101; 222; 298; 219; 31; 161; 47; 92; 64; 85; 124; 234; 168; 247; 227; 145; 262; 242; 274; 38; 24; 214; 282; 201; 205; 294; 83; 288; 151; 233; 87; 50; 29; 137; 228; 68; 142; 149; 271; 32; 221; 122; 23; 279; 172; 171; 63; 131; 139; 51; 88; 223; 291; 94; 292; 140; 256; 189; 152; 10; 127; 81; 251; 289; 167; 249; 215; 13; 61; 36; 55; 129; 192; 98; 66; 19; 30; 258; 286; 261; 264; 5; 143; 62; 185; 207; 8; 297; 40; 275; 239; 120; 45; 33; 128; 15; 226; 159; 39; 103; 188; 17; 113; 95; 96; 290; 174; 217; 126; 93; 108; 230; 238; 147; 4; 26; 216; 200; 209; 132; 79; 285; 56; 22; 211; 154; 72; 76; 138; 46; 164; 177; 283; 54; 115; 71; 163; 180; 206; 176; 280; 2; 28; 299; 287; 9; 69; 150; 44; 153; 196; 97; 102; 184; 243; 57; 252; 80; 148; 165; 112; 0; 248; 53; 284; 162; 91; 114; 254; 116; 244; 18; 232; 273; 43; 179; 197; 105; 157; 20; 266; 237; 235; 212; 183; 182; 121; 99; 84; 59; 58; 202; 156; 296; 195; 277; 70; 198; 170; 60; 175; 107; 263; 181; 245; 203; 65; 260; 225; 246; 25; 270; 187; 27; 110; 160; 240; 173; 73; 109; 123; 134; 208; 146; 119; 158; 100; 89; 82; 269; 204; 104; 74; 118; 141; 49; 278; 259; 130; 241; 253; 155|]
       [|114; 99; 198; 80; 30; 82; 51; 61; 58; 119; 228; 259; 97; 12; 13; 251; 174; 211; 131; 257; 79; 90; 288; 9; 100; 285; 281; 34; 216; 286; 46; 184; 125; 150; 151; 7; 263; 158; 168; 289; 112; 284; 219; 148; 270; 54; 157; 239; 248; 278; 71; 142; 136; 269; 220; 41; 183; 67; 208; 23; 68; 159; 52; 169; 267; 250; 43; 231; 192; 127; 185; 101; 218; 62; 241; 215; 237; 63; 152; 42; 18; 291; 5; 181; 78; 287; 109; 221; 107; 224; 205; 17; 196; 92; 143; 135; 0; 203; 140; 212; 276; 128; 40; 268; 264; 10; 70; 36; 277; 86; 106; 60; 73; 172; 33; 110; 65; 118; 232; 98; 260; 84; 28; 238; 39; 204; 209; 66; 64; 223; 93; 210; 38; 234; 275; 175; 235; 271; 163; 177; 266; 32; 124; 15; 117; 200; 161; 214; 155; 145; 76; 19; 242; 115; 282; 252; 96; 240; 162; 108; 44; 69; 197; 95; 6; 37; 129; 2; 21; 170; 102; 8; 296; 49; 94; 187; 186; 89; 14; 103; 226; 283; 31; 116; 27; 182; 122; 45; 77; 207; 88; 87; 132; 48; 111; 249; 105; 104; 56; 55; 206; 290; 202; 191; 274; 167; 254; 201; 194; 178; 137; 294; 29; 233; 297; 246; 3; 171; 83; 189; 134; 91; 4; 265; 188; 293; 299; 20; 35; 245; 50; 57; 195; 225; 279; 227; 164; 121; 53; 154; 247; 199; 26; 144; 193; 272; 25; 253; 133; 47; 141; 165; 72; 59; 156; 81; 130; 292; 173; 217; 85; 256; 261; 147; 222; 190; 138; 166; 243; 273; 262; 123; 113; 255; 24; 213; 179; 120; 180; 22; 11; 280; 146; 1; 75; 139; 149; 160; 258; 298; 229; 126; 295; 236; 153; 244; 230; 16; 74; 176|]
       [|64; 233; 215; 5; 91; 247; 273; 278; 134; 242; 94; 290; 172; 281; 152; 213; 271; 262; 150; 88; 147; 23; 158; 151; 221; 96; 66; 182; 272; 69; 153; 210; 222; 195; 227; 225; 44; 116; 20; 11; 165; 211; 60; 297; 264; 130; 293; 17; 201; 100; 70; 121; 244; 8; 3; 40; 102; 143; 235; 29; 111; 226; 112; 39; 15; 146; 24; 106; 161; 261; 148; 192; 183; 18; 274; 115; 241; 269; 6; 179; 198; 140; 47; 26; 16; 199; 117; 135; 51; 80; 283; 141; 124; 298; 35; 77; 10; 129; 296; 49; 105; 258; 87; 137; 230; 90; 82; 14; 203; 54; 108; 279; 46; 194; 31; 71; 204; 167; 250; 190; 125; 216; 219; 255; 142; 197; 168; 193; 284; 275; 53; 58; 128; 7; 126; 63; 118; 93; 200; 282; 122; 114; 89; 256; 68; 186; 170; 237; 217; 196; 175; 2; 245; 149; 240; 220; 160; 288; 73; 22; 109; 181; 36; 164; 162; 42; 157; 110; 236; 166; 174; 19; 214; 43; 171; 155; 176; 285; 34; 259; 223; 254; 189; 98; 136; 38; 28; 177; 291; 123; 202; 95; 270; 4; 263; 33; 55; 299; 76; 59; 50; 224; 252; 86; 99; 180; 72; 267; 56; 266; 212; 62; 277; 286; 209; 103; 289; 163; 48; 119; 83; 145; 85; 295; 12; 97; 131; 113; 228; 139; 1; 52; 229; 144; 178; 132; 78; 9; 188; 243; 92; 206; 185; 238; 74; 292; 276; 253; 21; 127; 169; 79; 184; 13; 280; 81; 205; 75; 57; 65; 30; 239; 45; 191; 104; 257; 246; 218; 27; 187; 265; 159; 251; 248; 173; 207; 67; 231; 232; 41; 101; 120; 37; 133; 84; 0; 287; 294; 260; 32; 25; 154; 61; 208; 138; 156; 234; 249; 268; 107|]
       [|202; 292; 91; 137; 152; 122; 178; 148; 79; 281; 194; 261; 277; 134; 38; 64; 259; 185; 258; 279; 175; 255; 164; 75; 142; 248; 132; 170; 183; 135; 71; 21; 111; 153; 231; 191; 31; 13; 190; 44; 9; 249; 47; 109; 7; 189; 211; 52; 273; 198; 97; 126; 30; 50; 86; 165; 80; 158; 223; 99; 241; 110; 20; 233; 256; 227; 24; 94; 230; 92; 58; 101; 11; 283; 49; 150; 98; 56; 140; 186; 55; 83; 128; 129; 228; 154; 36; 167; 124; 299; 280; 201; 284; 174; 254; 106; 252; 214; 67; 78; 219; 32; 271; 286; 145; 264; 93; 262; 176; 144; 294; 222; 251; 5; 6; 8; 285; 127; 12; 2; 295; 204; 296; 121; 59; 115; 131; 149; 274; 26; 250; 276; 239; 89; 95; 197; 162; 40; 200; 288; 297; 278; 267; 146; 43; 181; 244; 138; 206; 272; 215; 269; 193; 25; 173; 268; 218; 117; 209; 290; 159; 57; 107; 212; 72; 225; 187; 37; 179; 74; 105; 65; 139; 48; 287; 3; 298; 270; 141; 68; 0; 113; 253; 116; 216; 120; 125; 27; 188; 172; 63; 76; 275; 213; 61; 77; 171; 10; 235; 136; 60; 100; 234; 33; 217; 34; 161; 54; 242; 123; 208; 207; 147; 108; 195; 224; 169; 114; 210; 180; 257; 220; 82; 14; 35; 166; 46; 184; 289; 112; 73; 192; 182; 69; 238; 103; 163; 84; 28; 4; 70; 119; 160; 246; 282; 16; 247; 90; 18; 87; 226; 291; 151; 62; 51; 81; 42; 265; 236; 130; 88; 29; 53; 240; 45; 243; 229; 66; 263; 260; 104; 22; 168; 41; 232; 293; 96; 1; 177; 15; 102; 157; 205; 19; 39; 196; 143; 266; 23; 237; 17; 221; 118; 85; 203; 199; 155; 156; 133; 245|] |]

  let inversion = 
    if (X.ColumnCount = 300) then
      i <- i + 1
      parray.[i - 1]
    else
      // TODO: add epoch # to seed
      Combinatorics.GeneratePermutation(X.ColumnCount, Random(seed))

  let permutation = Permutation(inversion).Inverse()
  X.PermuteColumns(permutation)
  Y.PermuteColumns(permutation)

let private __trainNetworkFor1MiniBatch arch hp (J: double, parameters: Parameters, timer: Stopwatch) (X: Matrix<double>, Y: Matrix<double>): (double * Parameters * Stopwatch) =
  timer.Start()
  let Ŷ, caches = _forwardPropagateTrain arch parameters X
  let gradients = _backwardPropagate arch hp.λ Y Ŷ caches
  let parameters = _updateParameters arch hp.α parameters gradients
  timer.Stop()
  let m = X.ColumnCount
  let J = J + (double m) * (_computeCost hp.λ Y Ŷ parameters)
  J, parameters, timer

// TODO: pass seed separately and pass it on to shuffle
let trainNetwork paramsInit (callback: EpochCallback) (arch: Architecture) (X: Matrix<double>) (Y: Matrix<double>) (hp: HyperParameters): Parameters =
  assert (X.ColumnCount = Y.ColumnCount)
  let m = X.ColumnCount

  let timer = Stopwatch()
  // TODO: pull it out into a different method
  let _folder parameters epoch =
    timer.Restart()
    // NOTE: Should we move it out of there?
    // TODO: Ask on various forums what is the best practice if dataset is large
    // TODO: Do this at the start of trainNetwork
    let X = X.Clone()
    let Y = Y.Clone()
    shuffleDataSet_ 0 (X, Y)

    let J, parameters, timer = 
      (X, Y)
      |> _getMiniBatches hp.BatchSize
      |> Seq.fold (__trainNetworkFor1MiniBatch arch hp) (0., parameters, timer)
    timer.Stop()
    let J = J / double m
    let accuracy = computeAccuracy arch X Y parameters
    callback epoch timer.Elapsed J accuracy
    parameters

  let p0 = 
    match paramsInit with
    | Parameters ps -> ps
    | Seed s -> arch |> _initializeParameters hp.HeScale s

  seq { for epoch in 0 .. (hp.Epochs - 1) do epoch }
  |> Seq.fold _folder p0

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
