module YannLib.Tests.ActivationsTests

open global.Xunit
open YannLib.Activations
open YannLib.Tests.TestHelpers

[<Fact>]
let ``Check softmax forward``() =
  let x =
    [[1.; 0.; 3.; 5.]] |> toM

  let smx = Softmax.f x

  smx |> shouldBeEquivalentM [[0.01578405; 0.00580663; 0.11662925; 0.86178007]]

[<Fact>]
let ``Check softmax backward``() =
  let x =
    [[0.; 4.; 5.]] |> toM

  let smx = Softmax.df x

  smx |> shouldBeEquivalentM [[ 0.00487766; -0.00131181; -0.00356586]
                              [-0.00131181;  0.196001  ; -0.1946892 ]
                              [-0.00356586; -0.1946892 ;  0.19825505]]

