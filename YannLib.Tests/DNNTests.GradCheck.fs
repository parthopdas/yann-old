module YannLib.Tests.DNNTests.GradCheck

open global.Xunit
open YannLib.DNN
open FluentAssertions
open YannLib.Tests.TestHelpers

[<Fact>]
let ``Check Gradient check``() =
  let W1 = 
    [[-0.3224172 ; -0.38405435;  1.13376944; -1.09989127]
     [-0.17242821; -0.87785842;  0.04221375;  0.58281521]
     [-1.10061918;  1.14472371;  0.90159072;  0.50249434]
     [ 0.90085595; -0.68372786; -0.12289023; -0.93576943]
     [-0.26788808;  0.53035547; -0.69166075; -0.39675353]] |> toM
  let b1 = 
    [|-0.6871727 ; -0.84520564; -0.67124613; -0.0126646 ; -1.11731035|] |> toV
  let W2 = 
      [[ 0.2344157 ;  1.65980218;  0.74204416; -0.19183555; -0.88762896]
       [-0.74715829;  1.6924546 ;  0.05080775; -0.63699565;  0.19091548]
       [ 2.10025514;  0.12015895;  0.61720311;  0.30017032; -0.35224985]] |> toM
  let b2 = 
      [|-1.1425182 ; -0.34934272; -0.20889423|] |> toV
  let W3 = 
      [[0.58662319; 0.83898341; 0.93110208]] |> toM
  let b3 = 
      [|0.28558733|] |> toV

  let X = 
    [[ 1.62434536; -0.61175641; -0.52817175]
     [-1.07296862;  0.86540763; -2.3015387 ]
     [ 1.74481176; -0.7612069;   0.3190391 ]
     [-0.24937038;  1.46210794; -2.06014071]] |> toM  
  let Y = [[1.; 1.; 0.]] |> toM

  let parameters = [(1, { W = W1; b = b1 }); (2, { W = W2; b = b2 }); (3, { W = W3; b = b3 })] |> Map.ofList

  let arch =
    { nₓ = 4
      Layers =
        [ { n = 5; Activation = ReLU }
          { n = 3; Activation = ReLU }
          { n = 1; Activation = Sigmoid } ] }

  let ε = 1e-7
  let delta = calculateDeltaForGradientCheck 0. ε arch parameters X Y
  delta.Should().BeLessThan(1e-7, System.String.Empty, Array.empty) |> ignore
