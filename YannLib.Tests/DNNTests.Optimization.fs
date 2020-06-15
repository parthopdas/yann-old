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
      BatchSize = BatchSize64 }

  let cmap = Dictionary<int, double>()
  let callback =
    fun e _ J _ -> if e % 1 = 0 then cmap.[e] <- J else ()

  let p0 = DataLoaders.loadParameters 3 "data\\params.mat"
  let parameters = DNN.trainNetwork (DNN.Parameters p0) callback arch X Y hp

  let costs = cmap |> Seq.sortBy (fun kv -> kv.Key) |> Seq.map (fun kv -> kv.Value) |> Vector<double>.Build.DenseOfEnumerable
  costs |> shouldBeEquivalentV [| 0.702405; 0.702364; 0.702320; 0.702280; 0.702234 |]

  let accuracy = DNN.computeAccuracy arch X Y parameters
  //accuracy |> shouldBeApproximately 0.79666666
  accuracy |> shouldBeApproximately 0.60666666
  
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
