module YannLib.Tests.DNNTests.Optimization

open global.Xunit
open YannLib
open YannLib.DNN
open YannLib.Tests.TestHelpers
open MathNet.Numerics.Data.Matlab
open System.Collections.Generic
open MathNet.Numerics.LinearAlgebra
open FsUnit.Xunit

[<Fact>]
let ``Check cost with MBGD``() =
  let dataFile = [() |> Path.getExecutingAssemblyLocation; "data"; "optimization.moons.mat"] |> Path.combine
  let data = MatlabReader.ReadAll<double>(dataFile, "X", "Y")
  let X, Y = data.["X"], data.["Y"]

  let arch =
    { nₓ = X.RowCount
      Layers =
        [| { n = 5; Activation = ReLU; KeepProb = None }
           { n = 2; Activation = ReLU; KeepProb = None }
           { n = 1; Activation = Sigmoid; KeepProb = None } |] }

  let hp =
    { Epochs = 5
      α = 0.0007
      HeScale = 1.
      λ = None
      Optimization = NoOptimization
      BatchSize = BatchSize64 }

  let cmap = Dictionary<int, double>()
  let callback =
    fun e _ J _ -> if e % 1 = 0 then cmap.[e] <- J else ()

  let p0 = DataLoaders.loadParameters 3 "data\\params.mat"
  let parameters = DNN.trainNetwork (DNN.Parameters p0) callback arch hp X Y

  let costs = cmap |> Seq.sortBy (fun kv -> kv.Key) |> Seq.map (fun kv -> kv.Value) |> Vector<double>.Build.DenseOfEnumerable
  costs |> shouldBeEquivalentV [| 0.702405; 0.702364; 0.702320; 0.702280; 0.702234 |]

  let accuracy = DNN.computeAccuracy arch X Y parameters
  //accuracy |> shouldBeApproximately 0.79666666
  accuracy |> shouldBeApproximately 0.60666666

[<Fact>]
let ``Check update parameters with momentum``() =
  let W1 =
    [[ 1.62434536; -0.61175641; -0.52817175]
     [-1.07296862;  0.86540763; -2.3015387 ]] |> toM
  let b1 =
    [|1.74481176; -0.7612069|] |> toV 
  let W2 =  
    [[ 0.3190391 ; -0.24937038]
     [-2.06014071; -0.3224172 ]
     [ 1.13376944; -1.09989127]] |> toM 
  let b2 = 
    [|-0.87785842; 0.04221375; 0.58281521|] |> toV
  let parameters = [(1, { W = W1; b = b1 }); (2, { W = W2; b = b2 })] |> Map.ofList

  let dW1 = 
    [[-1.10061918;  1.14472371;  0.90159072]
     [ 0.50249434;  0.90085595; -0.68372786]] |> toM
  let db1 = 
    [|-0.12289023; -0.93576943|] |> toV 
  let dW2 =
    [[-0.26788808;  0.53035547]
     [-0.39675353; -0.6871727 ]
     [-0.67124613; -0.0126646 ]] |> toM 
  let db2 = 
    [|0.2344157; 1.65980218; 0.74204416|] |> toV
  let gradients = [(1, { _invalidGradient with dW = dW1; db = db1 }); (2, { _invalidGradient with dW = dW2; db = db2 })] |> Map.ofList

  let arch =
    { nₓ = 3
      Layers =
        [| { n = 2; Activation = ReLU; KeepProb = None }
           { n = 3; Activation = ReLU; KeepProb = None } |] }

  let v = _initializeGradientVelocities arch

  let ts = _updateParametersWithMomentum arch 0.01 MomentumParameters.Defaults gradients (parameters, v)
  let parameters, v =
    match ts with
    | MomentumTrainingState (p, v) -> p, v
    | _ -> Prelude.undefined

  parameters.[1].W |> shouldBeEquivalentM [[1.62544598; -0.61290114; -0.52907334]; [-1.07347112; 0.86450677; -2.30085497]]
  parameters.[1].b |> shouldBeEquivalentV [|1.74493465; -0.76027113 |]
  parameters.[2].W |> shouldBeEquivalentM [[0.31930698; -0.24990073]; [-2.05974396; -0.32173003]; [1.13444069; -1.0998786]]
  parameters.[2].b |> shouldBeEquivalentV [|-0.87809283; 0.04055394; 0.58207317 |]

  v.[1].dWv |> shouldBeEquivalentM [[-0.11006192; 0.11447237; 0.09015907]; [ 0.05024943; 0.09008559; -0.06837279]]
  v.[1].dbv |> shouldBeEquivalentV [|-0.01228902; -0.09357694|]
  v.[2].dWv |> shouldBeEquivalentM [[-0.02678881; 0.05303555]; [-0.03967535; -0.06871727]; [-0.06712461; -0.00126646]]
  v.[2].dbv |> shouldBeEquivalentV [|0.02344157; 0.16598022; 0.07420442|]

[<Fact>]
let ``Check cost with momentum``() =
  let dataFile = [() |> Path.getExecutingAssemblyLocation; "data"; "optimization.moons.mat"] |> Path.combine
  let data = MatlabReader.ReadAll<double>(dataFile, "X", "Y")
  let X, Y = data.["X"], data.["Y"]

  let arch =
    { nₓ = X.RowCount
      Layers =
        [| { n = 5; Activation = ReLU; KeepProb = None }
           { n = 2; Activation = ReLU; KeepProb = None }
           { n = 1; Activation = Sigmoid; KeepProb = None } |] }

  let hp =
    { Epochs = 5
      α = 0.0007
      HeScale = 1.
      λ = None
      Optimization = MomentumOptimization MomentumParameters.Defaults
      BatchSize = BatchSize64 }

  let cmap = Dictionary<int, double>()
  let callback =
    fun e _ J _ -> if e % 1 = 0 then cmap.[e] <- J else ()

  let p0 = DataLoaders.loadParameters 3 "data\\params.mat"
  let parameters = DNN.trainNetwork (DNN.Parameters p0) callback arch hp X Y

  let costs = cmap |> Seq.sortBy (fun kv -> kv.Key) |> Seq.map (fun kv -> kv.Value) |> Vector<double>.Build.DenseOfEnumerable
  costs |> shouldBeEquivalentV [| 0.702413; 0.702397; 0.702372; 0.702341; 0.702305 |]

  let accuracy = DNN.computeAccuracy arch X Y parameters
  //accuracy |> shouldBeApproximately 0.79666666
  accuracy |> shouldBeApproximately 0.60666666
  
[<Fact>]
let ``Check update parameters with ADAM``() =
  let W1p =
    [[ 1.62434536; -0.61175641; -0.52817175]
     [-1.07296862;  0.86540763; -2.3015387]] |> toM
  let b1p =
    [|1.74481176; -0.7612069|] |> toV 
  let W2p =  
    [[ 0.3190391 ; -0.24937038]
     [-2.06014071; -0.3224172]
     [ 1.13376944; -1.09989127]] |> toM 
  let b2p = 
    [|-0.87785842; 0.04221375; 0.58281521|] |> toV
  let parameters = [(1, { W = W1p; b = b1p }); (2, { W = W2p; b = b2p })] |> Map.ofList

  let dW1 =
    [[-1.10061918;  1.14472371;  0.90159072]
     [ 0.50249434;  0.90085595; -0.68372786]] |> toM
  let db1 = 
    [|-0.12289023; -0.93576943|] |> toV 
  let dW2 =
    [[-0.26788808;  0.53035547]
     [-0.39675353; -0.6871727 ]
     [-0.67124613; -0.0126646 ]] |> toM 
  let db2 = 
    [|0.2344157 ; 1.65980218; 0.74204416|] |> toV
  let gradients = [(1, { _invalidGradient with dW = dW1; db = db1 }); (2, { _invalidGradient with dW = dW2; db = db2 })] |> Map.ofList

  let arch =
    { nₓ = 3
      Layers =
        [| { n = 2; Activation = ReLU; KeepProb = None }
           { n = 3; Activation = ReLU; KeepProb = None } |] }

  let v, s = _initializeGradientVelocities arch, _initializeSquaredGradientVelocities arch

  let ts = _updateParametersWithADAM arch 0.01 ADAMParameters.Defaults gradients (parameters, v, s, 2.)
  let parameters, v, s, t =
    match ts with
    | ADAMTrainingState (p, v, s, t) -> p, v, s, t
    | _ -> Prelude.undefined

  parameters.[1].W |> shouldBeEquivalentM [[1.63178673; -0.61919778; -0.53561312]; [-1.08040999;  0.85796626; -2.29409733]]
  parameters.[1].b |> shouldBeEquivalentV [|1.75225313; -0.75376553|]
  parameters.[2].W |> shouldBeEquivalentM [[0.32648046; -0.25681174]; [-2.05269934; -0.31497584]; [1.14121081; -1.09244991]]
  parameters.[2].b |> shouldBeEquivalentV [|-0.88529979; 0.03477238; 0.57537385 |]

  v.[1].dWv |> shouldBeEquivalentM [[-0.11006192; 0.11447237; 0.09015907]; [0.05024943; 0.09008559; -0.06837279]]
  v.[1].dbv |> shouldBeEquivalentV [|-0.01228902; -0.09357694|]
  v.[2].dWv |> shouldBeEquivalentM [[-0.02678881; 0.05303555]; [-0.03967535; -0.06871727]; [-0.06712461; -0.00126646]]
  v.[2].dbv |> shouldBeEquivalentV [|0.02344157; 0.16598022; 0.07420442|]

  s.[1].dWs |> shouldBeEquivalentM [[0.00121136; 0.00131039; 0.00081287]; [0.0002525; 0.00081154; 0.00046748]]
  s.[1].dbs |> shouldBeEquivalentV [|1.51020075e-05; 8.75664434e-04|]
  s.[2].dWs |> shouldBeEquivalentM [[7.17640232e-05; 2.81276921e-04]; [1.57413361e-04; 4.72206320e-04]; [4.50571368e-04; 1.60392066e-07]]
  s.[2].dbs |> shouldBeEquivalentV [|5.49507194e-05; 2.75494327e-03; 5.50629536e-04|]

  t |> shouldBeApproximately 3.

[<Fact>]
let ``Check cost with ADAM``() =
  let dataFile = [() |> Path.getExecutingAssemblyLocation; "data"; "optimization.moons.mat"] |> Path.combine
  let data = MatlabReader.ReadAll<double>(dataFile, "X", "Y")
  let X, Y = data.["X"], data.["Y"]

  let arch =
    { nₓ = X.RowCount
      Layers =
        [| { n = 5; Activation = ReLU; KeepProb = None }
           { n = 2; Activation = ReLU; KeepProb = None }
           { n = 1; Activation = Sigmoid; KeepProb = None } |] }

  let hp =
    { Epochs = 5
      α = 0.0007
      HeScale = 1.
      λ = None
      Optimization = ADAMOptimization ADAMParameters.Defaults
      BatchSize = BatchSize64 }

  let cmap = Dictionary<int, double>()
  let callback =
    fun e _ J _ -> if e % 1 = 0 then cmap.[e] <- J else ()

  let p0 = DataLoaders.loadParameters 3 "data\\params.mat"
  let parameters = DNN.trainNetwork (DNN.Parameters p0) callback arch hp X Y

  let costs = cmap |> Seq.sortBy (fun kv -> kv.Key) |> Seq.map (fun kv -> kv.Value) |> Vector<double>.Build.DenseOfEnumerable
  costs |> shouldBeEquivalentV [| 0.702166; 0.700860; 0.699807; 0.698633; 0.697517 |]

  let accuracy = DNN.computeAccuracy arch X Y parameters
  //accuracy |> shouldBeApproximately 0.79666666
  accuracy |> shouldBeApproximately 0.61666666

[<Fact>]
let ``Check _getMiniBatches for 1``() =
  let X = DenseMatrix.init 100 100 (fun i j -> float i * 10.0 + float j)
  let Y = DenseMatrix.init 4   100 (fun i j -> float i * 20.0 + float j)
  let batches = DNN._getMiniBatches BatchSize1 (X, Y)

  batches |> Seq.length |> should equal 100

  let X0, Y0 = batches |> Seq.skip 27 |> Seq.take 1 |> Seq.exactlyOne
  X0 |> shouldBeEquivalentM2 (DenseMatrix.init 100 1 (fun i j -> float i * 10.0 + 27.) |> Matrix.toArray2)
  Y0 |> shouldBeEquivalentM [[27.]; [47.]; [67.]; [87.]]

  let X0, Y0 = batches |> Seq.skip 99 |> Seq.take 1 |> Seq.exactlyOne
  X0 |> shouldBeEquivalentM2 (DenseMatrix.init 100 1 (fun i _ -> float i * 10.0 + 99.) |> Matrix.toArray2)
  Y0 |> shouldBeEquivalentM [[99.]; [119.]; [139.]; [159.]]

[<Fact>]
let ``Check _getMiniBatches - complete``() =
  let X = DenseMatrix.init 10 128 (fun i j -> float i * 10.0 + float j)
  let Y = DenseMatrix.init 3  128 (fun i j -> float i * 20.0 + float j)
  let batches = DNN._getMiniBatches BatchSize64 (X, Y)

  batches |> Seq.length |> should equal 2

  let X0, Y0 = batches |> Seq.skip 0 |> Seq.take 1 |> Seq.exactlyOne
  X0 |> shouldBeEquivalentM2 (DenseMatrix.init 10 64 (fun i j -> float i * 10.0 + float j) |> Matrix.toArray2)
  Y0 |> shouldBeEquivalentM2 (DenseMatrix.init 3  64 (fun i j -> float i * 20.0 + float j) |> Matrix.toArray2)

  let X0, Y0 = batches |> Seq.skip 1 |> Seq.take 1 |> Seq.exactlyOne
  X0 |> shouldBeEquivalentM2 (DenseMatrix.init 10 64 (fun i j -> float i * 10.0 + float j + 64.) |> Matrix.toArray2)
  Y0 |> shouldBeEquivalentM2 (DenseMatrix.init 3  64 (fun i j -> float i * 20.0 + float j + 64.) |> Matrix.toArray2)

[<Fact>]
let ``Check _getMiniBatches - partial``() =
  let X = DenseMatrix.init 11 100 (fun i j -> float i * 10.0 + float j)
  let Y = DenseMatrix.init 5  100 (fun i j -> float i * 20.0 + float j)
  let batches = DNN._getMiniBatches BatchSize128 (X, Y)

  batches |> Seq.length |> should equal 1

  let X0, Y0 = batches |> Seq.skip 0 |> Seq.take 1 |> Seq.exactlyOne
  X0 |> shouldBeEquivalentM2 (DenseMatrix.init 11 100 (fun i j -> float i * 10.0 + float j) |> Matrix.toArray2)
  Y0 |> shouldBeEquivalentM2 (DenseMatrix.init 5  100 (fun i j -> float i * 20.0 + float j) |> Matrix.toArray2)

[<Fact>]
let ``Check _getMiniBatches - complete and partial``() =
  let X = DenseMatrix.init 11 300 (fun i j -> float i * 10.0 + float j)
  let Y = DenseMatrix.init 5  300 (fun i j -> float i * 20.0 + float j)
  let batches = DNN._getMiniBatches BatchSize256 (X, Y)

  batches |> Seq.length |> should equal 2

  let X0, Y0 = batches |> Seq.skip 0 |> Seq.take 1 |> Seq.exactlyOne
  X0 |> shouldBeEquivalentM2 (DenseMatrix.init 11 256 (fun i j -> float i * 10.0 + float j) |> Matrix.toArray2)
  Y0 |> shouldBeEquivalentM2 (DenseMatrix.init 5  256 (fun i j -> float i * 20.0 + float j) |> Matrix.toArray2)

  let X1, Y1 = batches |> Seq.skip 1 |> Seq.take 1 |> Seq.exactlyOne
  X1 |> shouldBeEquivalentM2 (DenseMatrix.init 11 44 (fun i j -> float i * 10.0 + float j + 256.) |> Matrix.toArray2)
  Y1 |> shouldBeEquivalentM2 (DenseMatrix.init 5  44 (fun i j -> float i * 20.0 + float j + 256.) |> Matrix.toArray2)

[<Fact>]
let ``Check _getMiniBatches - all``() =
  let X = DenseMatrix.init 17 64 (fun i j -> float i * 10.0 + float j)
  let Y = DenseMatrix.init 3  64 (fun i j -> float i * 20.0 + float j)
  let batches = DNN._getMiniBatches BatchSizeAll (X, Y)

  batches |> Seq.length |> should equal 1

  let X0, Y0 = batches |> Seq.skip 0 |> Seq.take 1 |> Seq.exactlyOne
  X0 |> shouldBeEquivalentM2 (DenseMatrix.init 17 64 (fun i j -> float i * 10.0 + float j) |> Matrix.toArray2)
  Y0 |> shouldBeEquivalentM2 (DenseMatrix.init 3  64 (fun i j -> float i * 20.0 + float j) |> Matrix.toArray2)
