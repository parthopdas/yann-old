module YannLib.Tests.CoreTests

open global.Xunit
open YannLib.Core
open FluentAssertions
open FsUnit.Xunit
open MathNet.Numerics.LinearAlgebra

[<Fact>]
let ``Check network initialization``() =
  let arch =
    { nₓ = 2
      Layers =
        [ { n = 3; Activation = ReLU }
          { n = 1; Activation = Sigmoid } ] }

  let nn = arch |> _initializeNetwork 1

  nn.Architecture |> should equal arch
  nn.Parameters.Keys.Should().Equal(1, 2) |> ignore

  let W1 =
    [[-0.002993466474; 0.01541654033]
     [-0.00463620853; 0.01904072042]
     [-0.001872509079; -0.0081251694]]
  nn.Parameters.[1].W |> TestHelpers.shouldBeEquivalentM W1

  let b1 =
    [|-0.002993466474; -0.00463620853; -0.001872509079|]
  nn.Parameters.[1].b |> TestHelpers.shouldBeEquivalentV b1

  let W2 =
    [[-0.002993466474; -0.00463620853; -0.001872509079]]
  nn.Parameters.[2].W |> TestHelpers.shouldBeEquivalentM W2

  let b2 =
    [|-0.002993466474|]
  nn.Parameters.[2].b |> TestHelpers.shouldBeEquivalentV b2

[<Fact>]
let ``Check linear part of a layer's forward propagation``() =
  let A =
    [[ 1.62434536; -0.61175641]
     [-0.52817175; -1.07296862]
     [ 0.86540763; -2.3015387 ]] |> array2D |> CreateMatrix.DenseOfArray
  let W =
    [[ 1.74481176; -0.7612069; 0.3190391 ]] |> array2D |> CreateMatrix.DenseOfArray
  let b =
    [|-0.24937038|] |> CreateVector.DenseOfArray

  let (Z, _) = _linearForward A W b
  Z |> TestHelpers.shouldBeEquivalentM [[ 3.26295337; -1.23429987]]

[<Fact>]
let ``Check linear and activation part of a layer's forward propagation``() =
  let Aprev =
    [[-0.41675785; -0.05626683]
     [-2.1361961;   1.64027081]
     [-1.79343559; -0.84174737]] |> array2D |> CreateMatrix.DenseOfArray
  let W =
    [[0.50288142; -1.24528809; -1.05795222]] |> array2D |> CreateMatrix.DenseOfArray
  let b =
    [|-0.90900761|] |> CreateVector.DenseOfArray

  let (A, _) = _linearActivationForward Aprev W b ReLU
  A |> TestHelpers.shouldBeEquivalentM [[ 3.43896131; 0.0 ]]

  let (A, _) = _linearActivationForward Aprev W b Sigmoid
  A |> TestHelpers.shouldBeEquivalentM [[ 0.96890023; 0.11013289 ]]

[<Fact>]
let ``Check full forward propagation``() =
  let X = 
    [[-0.31178367;  0.72900392;  0.21782079; -0.8990918 ]
     [-2.48678065;  0.91325152;  1.12706373; -1.51409323]
     [ 1.63929108; -0.4298936;   2.63128056;  0.60182225]
     [-0.33588161;  1.23773784;  0.11112817;  0.12915125]
     [ 0.07612761; -0.15512816;  0.63422534;  0.810655  ]] |> array2D |> CreateMatrix.DenseOfArray

  let W1 =
    [[ 0.35480861;  1.81259031; -1.3564758 ; -0.46363197;  0.82465384]
     [-1.17643148;  1.56448966;  0.71270509; -0.1810066 ;  0.53419953]
     [-0.58661296; -1.48185327;  0.85724762;  0.94309899;  0.11444143]
     [-0.02195668; -2.12714455; -0.83440747; -0.46550831;  0.23371059]] |> array2D |> CreateMatrix.DenseOfArray
  let b1 =
    [| 1.38503523; -0.51962709; -0.78015214; 0.95560959 |] |> CreateVector.DenseOfArray

  let W2 =
    [[-0.12673638; -1.36861282;  1.21848065; -0.85750144]
     [-0.56147088; -1.0335199 ;  0.35877096;  1.07368134]
     [-0.37550472;  0.39636757; -0.47144628;  2.33660781]] |> array2D |> CreateMatrix.DenseOfArray
  let b2 = [| 1.50278553; -0.59545972; 0.52834106 |] |> CreateVector.DenseOfArray

  let W3 =
    [[ 0.9398248 ;  0.42628539; -0.75815703]] |> array2D |> CreateMatrix.DenseOfArray
  let b3 =
    [|-0.16236698|] |> CreateVector.DenseOfArray

  let arch =
    { nₓ = 5
      Layers =
        [ { n = 4; Activation = ReLU }
          { n = 3; Activation = ReLU }
          { n = 1; Activation = Sigmoid } ] }
  let network =
    { Architecture = arch
      Parameters = dict [(1, {| W = W1; b = b1|}); (2, {| W = W2; b = b2|}); (3, {| W = W3; b = b3|})] }

  let AL, caches = _forwardPropagate network X
  
  caches.Count |> should equal 3
  let expected = [[ 0.03921668; 0.70498921; 0.19734387; 0.04728177]]
  AL |> TestHelpers.shouldBeEquivalentM expected
