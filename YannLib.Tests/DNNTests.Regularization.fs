module YannLib.Tests.DNNTests.Regularization

open global.Xunit
open YannLib.DNN
open YannLib.Tests.TestHelpers
open MathNet.Numerics.Data.Matlab
open System.Collections.Generic
open YannLib

[<Fact>]
let ``Check cost with regularization``() =
  let W1 =
    [[ 1.62434536; -0.61175641; -0.52817175]
     [-1.07296862;  0.86540763; -2.3015387]] |> toM
  let b1 =
    [| 1.74481176; -0.7612069 |] |> toV

  let W2 =
    [[ 0.3190391;  -0.24937038]
     [ 1.46210794; -2.06014071]
     [-0.3224172;  -0.38405435]] |> toM
  let b2 = [| 1.13376944; -1.09989127; -0.17242821 |] |> toV

  let W3 =
    [[ -0.87785842;  0.04221375;  0.58281521 ]] |> toM
  let b3 =
    [|-1.10061918|] |> toV

  let parameters = Map.ofList [(1, { W = W1; b = b1 }); (2, { W = W2; b = b2 }); (3, { W = W3; b = b3 })]

  let Y = [[1.; 1.; 0.; 1.; 0.]] |> toM
  let Ŷ = [[0.40682402;  0.01629284;  0.16722898;  0.10118111;  0.40682402]] |> toM

  let cost = _computeCost (Some 0.1) Y Ŷ parameters

  cost |> shouldBeApproximately 1.78648594516

[<Fact>]
let ``Check backward propagation with regularization``() =
  let arch =
    { nₓ = 3
      Layers =
        [| { n = 2; Activation = ReLU; KeepProb = None }
           { n = 3; Activation = ReLU; KeepProb = None  }
           { n = 1; Activation = Sigmoid; KeepProb = None } |] }
  let X =
    [[ 1.62434536; -0.61175641; -0.52817175; -1.07296862;  0.86540763]
     [-2.3015387 ;  1.74481176; -0.7612069 ;  0.3190391 ; -0.24937038]
     [ 1.46210794; -2.06014071; -0.3224172 ; -0.38405435;  1.13376944]] |> toM
  let Z1 =
    [[-1.52855314;  3.32524635;  2.13994541;  2.60700654; -0.75942115]
     [-1.98043538;  4.1600994 ;  0.79051021;  1.46493512; -0.45506242]] |> toM
  let A1 =
    [[ 0.        ;  3.32524635;  2.13994541;  2.60700654;  0.        ]
     [ 0.        ;  4.1600994 ;  0.79051021;  1.46493512;  0.        ]] |> toM
  let W1 =
    [[-1.09989127; -0.17242821; -0.87785842]
     [ 0.04221375;  0.58281521; -1.10061918]] |> toM
  let b1 =
    [| 1.14472371; 0.90159072 |] |> toV

  let Z2 =
    [[ 0.53035547;  5.94892323;  2.31780174;  3.16005701;  0.53035547]
     [-0.69166075; -3.47645987; -2.25194702; -2.65416996; -0.69166075]
     [-0.39675353; -4.62285846; -2.61101729; -3.22874921; -0.39675353]] |> toM
  let A2 =
    [[ 0.53035547;  5.94892323;  2.31780174;  3.16005701;  0.53035547]
     [ 0.        ;  0.        ;  0.        ;  0.        ;  0.        ]
     [ 0.        ;  0.        ;  0.        ;  0.        ;  0.        ]] |> toM
  let W2 =
    [[ 0.50249434;  0.90085595]    
     [-0.68372786; -0.12289023]
     [-0.93576943; -0.26788808]] |> toM
  let b2 = [| 0.53035547; -0.69166075; -0.39675353 |] |> toV

  let Z3 =
    [[-0.3771104 ; -4.10060224; -1.60539468; -2.18416951; -0.3771104 ]] |> toM
  let A3 =
    [[ 0.40682402;  0.01629284;  0.16722898;  0.10118111;  0.40682402]] |> toM
  let W3 =
    [[ -0.6871727 ; -0.84520564; -0.67124613 ]] |> toM
  let b3 =
    [|-0.0126646|] |> toV

  let Y = [[1.; 1.; 0.; 1.; 0.]] |> toM
  let Ŷ = [[ 0.40682402;  0.01629284;  0.16722898;  0.10118111;  0.40682402]] |> toM

  let caches =
    [(1, { Aprev = X; D = None; Z = Z1; W = W1; b = b1 })
     (2, { Aprev = A1; D = None; Z = Z2; W = W2; b = b2 })
     (3, { Aprev = A2; D = None; Z = Z3; W = W3; b = b3 })] |> Map.ofList

  let dW1 =
    [[-0.25604646;  0.12298827; -0.28297129]
     [-0.17706303;  0.34536094; -0.4410571 ]]
  let dW2 =
    [[ 0.79276486;  0.85133918]
     [-0.0957219 ; -0.01720463]
     [-0.13100772; -0.03750433]]
  let dW3 =
    [[-1.77691347; -0.11832879; -0.09397446]]
  let db1 =
    [| 0.11845855;  0.21236874 |]
  let db2 =
    [| 0.26135226; 0.        ; 0.        |]
  let db3 =
    [|-0.38032981|]

  let grads = _backwardPropagate arch (Some 0.7) Y Ŷ caches

  grads.[1].dW |> shouldBeEquivalentM dW1
  grads.[1].db |> shouldBeEquivalentV db1
  grads.[2].dW |> shouldBeEquivalentM dW2
  grads.[2].db |> shouldBeEquivalentV db2
  grads.[3].dW |> shouldBeEquivalentM dW3
  grads.[3].db |> shouldBeEquivalentV db3

[<Fact>]
let ``Check predictions with regularization``() =
  MathNet.Numerics.Control.UseNativeMKL()

  let dataFile = [() |> Path.getExecutingAssemblyLocation; "data"; "regularization.traindev.mat"] |> Path.combine
  let data = MatlabReader.ReadAll<double>(dataFile, "X", "y", "Xval", "yval");
  let train_X = data.["X"].Transpose()
  let train_Y = data.["y"].Transpose()
  let test_X = data.["Xval"].Transpose()
  let test_Y = data.["yval"].Transpose()

  let arch =
    { nₓ = 2
      Layers =
        [| { n = 20; Activation = ReLU; KeepProb = None }
           { n = 3; Activation = ReLU; KeepProb = None }
           { n = 1; Activation = Sigmoid; KeepProb = None } |] }

  let hp =
    { Epochs = 3_000
      α = 0.3
      λ = Some 0.7 }

  let costs = Dictionary<int, double>()
  let callback =
    fun e _ J _ -> if e % 1000 = 0 then costs.[e] <- J else ()

  let ps0 = DataLoaders.loadParameters 3 "data\\regularization.ps0.mat"

  let parameters = DNN.trainNetwork (Parameters ps0) callback arch train_X train_Y hp

  let trainAccuracy = DNN.computeAccuracy arch train_X train_Y parameters
  trainAccuracy |> shouldBeApproximately 0.9289099

  let testAccuracy = DNN.computeAccuracy arch test_X test_Y parameters
  testAccuracy |> shouldBeApproximately 0.92999999

[<Fact>]
let ``Check forward propagation with dropout``() =
  let arch =
    { nₓ = 3
      Layers =
        [| { n = 2; Activation = ReLU; KeepProb = Some 0.7 }
           { n = 3; Activation = ReLU; KeepProb = Some 0.7 }
           { n = 1; Activation = Sigmoid; KeepProb = None } |] }
  let X_assess =
    [[ 1.62434536; -0.61175641; -0.52817175; -1.07296862;  0.86540763]
     [-2.3015387 ;  1.74481176; -0.7612069 ;  0.3190391 ; -0.24937038]
     [ 1.46210794; -2.06014071; -0.3224172 ; -0.38405435;  1.13376944]] |> toM
  let W1 =
    [[-1.09989127; -0.17242821; -0.87785842]
     [ 0.04221375;  0.58281521; -1.10061918]] |> toM
  let b1 =
    [|1.14472371; 0.90159072|] |> toV
  let W2 =
    [[ 0.50249434;  0.90085595]
     [-0.68372786; -0.12289023]
     [-0.93576943; -0.26788808]] |> toM
  let b2 =
    [| 0.53035547; -0.69166075; -0.39675353|] |> toV
  let W3 =
    [[-0.6871727 ; -0.84520564; -0.67124613]] |> toM
  let b3 =
    [|-0.0126646|] |> toV

  let parameters = Map.ofList [(1, { W = W1; b = b1 }); (2, { W = W2; b = b2 }); (3, { W = W3; b = b3 })]

  let AL, _ = _forwardPropagateTrain arch parameters X_assess

  AL |> shouldBeEquivalentM [[0.49683389; 0.05332326; 0.04565099; 0.49683389; 0.36974721]]

[<Fact>]
let ``Check backward propagation with dropout``() =
  let arch =
    { nₓ = 3
      Layers =
        [| { n = 2; Activation = ReLU; KeepProb = Some 0.8 }
           { n = 3; Activation = ReLU; KeepProb = Some 0.8  }
           { n = 1; Activation = Sigmoid; KeepProb = None  } |] }
  let X =
    [[ 1.62434536; -0.61175641; -0.52817175; -1.07296862;  0.86540763]
     [-2.3015387 ;  1.74481176; -0.7612069 ;  0.3190391 ; -0.24937038]
     [ 1.46210794; -2.06014071; -0.3224172 ; -0.38405435;  1.13376944]] |> toM
  let Z1 =
    [[-1.52855314;  3.32524635;  2.13994541;  2.60700654; -0.75942115]
     [-1.98043538;  4.1600994 ;  0.79051021;  1.46493512; -0.45506242]] |> toM
  let D1 =
    [[ 1.; 0.;  1.;  1.;  1.]
     [ 1.;  1.;  1.;  1.; 0.]] |> toM |> Some
  let A1 =
    [[ 0.        ;  0.        ;  4.27989081;  5.21401307;  0.        ]
     [ 0.        ;  8.32019881;  1.58102041;  2.92987024;  0.        ]] |> toM
  let W1 =
    [[-1.09989127; -0.17242821; -0.87785842]
     [ 0.04221375;  0.58281521; -1.10061918]] |> toM
  let b1 =
    [| 1.14472371; 0.90159072|] |> toV
  let Z2 =
    [[ 0.53035547;  8.02565606;  4.10524802;  5.78975856;  0.53035547]
     [-0.69166075; -1.71413186; -3.81223329; -4.61667916; -0.69166075]
     [-0.39675353; -2.62563561; -4.82528105; -6.0607449 ; -0.39675353]] |> toM
  let D2 =
    [[ 1.; 0.;  1.; 0.;  1.]
     [0.;  1.; 0.;  1.;  1.]
     [0.; 0.;  1.; 0.; 0.]] |> toM |> Some
  let A2 =
    [[ 1.06071093;  0.        ;  8.21049603;  0.        ;  1.06071093]
     [ 0.        ;  0.        ;  0.        ;  0.        ;  0.        ]
     [ 0.        ;  0.        ;  0.        ;  0.        ;  0.        ]] |> toM
  let W2 =
    [[ 0.50249434;  0.90085595]
     [-0.68372786; -0.12289023]
     [-0.93576943; -0.26788808]] |> toM
  let b2 =
    [| 0.53035547; -0.69166075; -0.39675353|] |> toV
  let Z3 =
    [[-0.7415562 ; -0.0126646 ; -5.65469333; -0.0126646 ; -0.7415562 ]] |> toM
  let D3 = None
  let A3 =
    [[ 0.32266394;  0.49683389;  0.00348883;  0.49683389;  0.32266394]] |> toM
  let W3 =
    [[-0.6871727 ; -0.84520564; -0.67124613]] |> toM
  let b3 =
    [|-0.0126646|] |> toV

  let Y = [[1.; 1.; 0.; 1.; 0.]] |> toM

  let caches =
    [(0, { _invalidCache with D = None })
     (1, { Aprev = X; D = D1; Z = Z1; W = W1; b = b1 })
     (2, { Aprev = A1; D = D2; Z = Z2; W = W2; b = b2 })
     (3, { Aprev = A2; D = D3; Z = Z3; W = W3; b = b3 })] |> Map.ofList

  let dA1 =
    [[ 0.36544439;  0.;         -0.00188233;  0.;         -0.17408748]
     [ 0.65515713;  0.;         -0.00337459;  0.;         -0.        ]]
  let dA2 =
    [[ 0.58180856;  0.        ; -0.00299679;  0.        ; -0.27715731]
     [ 0.        ;  0.53159854; -0.        ;  0.53159854; -0.34089673]
     [ 0.        ;  0.        ; -0.00292733;  0.        ; -0.        ]]

  let grads = _backwardPropagate arch None Y A3 caches

  grads.[1].dA |> shouldBeEquivalentM dA1
  grads.[2].dA |> shouldBeEquivalentM dA2

[<Fact>]
let ``Check predictions with dropout``() =
  MathNet.Numerics.Control.UseNativeMKL()

  let dataFile = [() |> Path.getExecutingAssemblyLocation; "data"; "regularization.traindev.mat"] |> Path.combine
  let data = MatlabReader.ReadAll<double>(dataFile, "X", "y", "Xval", "yval");
  let train_X = data.["X"].Transpose()
  let train_Y = data.["y"].Transpose()
  let test_X = data.["Xval"].Transpose()
  let test_Y = data.["yval"].Transpose()

  let arch =
    { nₓ = 2
      Layers =
        [| { n = 20; Activation = ReLU; KeepProb = Some 0.86 }
           { n = 3; Activation = ReLU; KeepProb = Some 0.86 }
           { n = 1; Activation = Sigmoid; KeepProb = None } |] }

  let hp =
    { Epochs = 3_000
      α = 0.3
      λ = None }

  let costs = Dictionary<int, double>()
  let callback =
    fun e _ J _ -> if e % 1000 = 0 then costs.[e] <- J else ()

  let ps0 = DataLoaders.loadParameters 3 "data\\regularization.ps0.mat"

  let parameters = DNN.trainNetwork (Parameters ps0) callback arch train_X train_Y hp

  let trainAccuracy = DNN.computeAccuracy arch train_X train_Y parameters
  trainAccuracy |> shouldBeApproximately 0.92417061

  let testAccuracy = DNN.computeAccuracy arch test_X test_Y parameters
  testAccuracy |> shouldBeApproximately 0.92999999

