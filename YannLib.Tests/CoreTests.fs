module YannLib.Tests.CoreTests

open global.Xunit
open YannLib.Core
open FluentAssertions
open FsUnit.Xunit
open MathNet.Numerics.LinearAlgebra
open System
open System.Collections.Generic

let private matrixShape (m: Matrix<double>) =
  m.RowCount, m.ColumnCount

let private compareArrays (a1: double[]) (a2: double[]) =
  Seq.zip a1 a2
  |> Seq.map (fun (e1, e2) -> Math.Abs(e1 - e2))
  |> Seq.filter ((<) 0.00000001)
  |> should be Empty
  

[<Fact>]
let ``Check network initialization``() =
  let arch = {
    n_x = 2
    Layers = [|
      FullyConnected {| n = 3; Activation = ReLU |}
      FullyConnected {| n = 1; Activation = Sigmoid |}
    |]
  }

  let nn = arch |> initializeNetwork 1

  nn.Parameters.Keys.Should().Equal(1, 2) |> ignore
  nn.Parameters.[1].W |> matrixShape |> should equal (3, 2)
  compareArrays (nn.Parameters.[1].W.ToRowMajorArray()) [|-0.002993466474; 0.01541654033; -0.00463620853; 0.01904072042; -0.001872509079; -0.0081251694|]
  nn.Parameters.[1].b |> matrixShape |> should equal (3, 1)
  compareArrays (nn.Parameters.[1].b.ToRowMajorArray()) [|-0.002993466474; -0.00463620853; -0.001872509079|]
  nn.Parameters.[2].W |> matrixShape |> should equal (1, 3)
  compareArrays (nn.Parameters.[2].W.ToRowMajorArray()) [|-0.002993466474; -0.00463620853; -0.001872509079|]
  nn.Parameters.[2].b |> matrixShape |> should equal (1, 1)
  compareArrays (nn.Parameters.[2].b.ToRowMajorArray()) [|-0.002993466474|]
