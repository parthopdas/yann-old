module YannLib.Tests.CoreTests

open global.Xunit
open YannLib.Core
open FluentAssertions
open FsUnit.Xunit
open TestHelpers

[<Fact>]
let ``Check network initialization``() =
  let arch =
    { nₓ = 2
      Layers =
        [ { n = 3; Activation = ReLU }
          { n = 1; Activation = Sigmoid } ] }

  let parameters = arch |> _initializeParameters 1

  parameters |> Map.toList |> List.map fst |> should equal [1; 2]

  let W1 =
    [[-0.002993466474; 0.01541654033]
     [-0.00463620853; 0.01904072042]
     [-0.001872509079; -0.0081251694]]
  parameters.[1].W |> shouldBeEquivalentM W1

  let b1 =
    [|-0.002993466474; -0.00463620853; -0.001872509079|]
  parameters.[1].b |> shouldBeEquivalentV b1

  let W2 =
    [[-0.002993466474; -0.00463620853; -0.001872509079]]
  parameters.[2].W |> shouldBeEquivalentM W2

  let b2 =
    [|-0.002993466474|]
  parameters.[2].b |> shouldBeEquivalentV b2

[<Fact>]
let ``Check linear part of a layer's forward propagation``() =
  let A =
    [[ 1.62434536; -0.61175641]
     [-0.52817175; -1.07296862]
     [ 0.86540763; -2.3015387 ]] |> toM
  let W =
    [[ 1.74481176; -0.7612069; 0.3190391 ]] |> toM
  let b =
    [|-0.24937038|] |> toV

  let cache = _linearForward A W b
  cache.Z |> shouldBeEquivalentM [[ 3.26295337; -1.23429987]]

[<Fact>]
let ``Check linear and activation part of a layer's forward propagation``() =
  let Aprev =
    [[-0.41675785; -0.05626683]
     [-2.1361961;   1.64027081]
     [-1.79343559; -0.84174737]] |> toM
  let W =
    [[0.50288142; -1.24528809; -1.05795222]] |> toM
  let b =
    [|-0.90900761|] |> toV

  let (A, _) = _linearActivationForward Aprev W b ReLU
  A |> shouldBeEquivalentM [[ 3.43896131; 0.0 ]]

  let (A, _) = _linearActivationForward Aprev W b Sigmoid
  A |> shouldBeEquivalentM [[ 0.96890023; 0.11013289 ]]

[<Fact>]
let ``Check full forward propagation``() =
  let X = 
    [[-0.31178367;  0.72900392;  0.21782079; -0.8990918 ]
     [-2.48678065;  0.91325152;  1.12706373; -1.51409323]
     [ 1.63929108; -0.4298936;   2.63128056;  0.60182225]
     [-0.33588161;  1.23773784;  0.11112817;  0.12915125]
     [ 0.07612761; -0.15512816;  0.63422534;  0.810655  ]] |> toM

  let W1 =
    [[ 0.35480861;  1.81259031; -1.3564758 ; -0.46363197;  0.82465384]
     [-1.17643148;  1.56448966;  0.71270509; -0.1810066 ;  0.53419953]
     [-0.58661296; -1.48185327;  0.85724762;  0.94309899;  0.11444143]
     [-0.02195668; -2.12714455; -0.83440747; -0.46550831;  0.23371059]] |> toM
  let b1 =
    [| 1.38503523; -0.51962709; -0.78015214; 0.95560959 |] |> toV

  let W2 =
    [[-0.12673638; -1.36861282;  1.21848065; -0.85750144]
     [-0.56147088; -1.0335199 ;  0.35877096;  1.07368134]
     [-0.37550472;  0.39636757; -0.47144628;  2.33660781]] |> toM
  let b2 = [| 1.50278553; -0.59545972; 0.52834106 |] |> toV

  let W3 =
    [[ 0.9398248 ;  0.42628539; -0.75815703]] |> toM
  let b3 =
    [|-0.16236698|] |> toV

  let arch =
    { nₓ = 5
      Layers =
        [ { n = 4; Activation = ReLU }
          { n = 3; Activation = ReLU }
          { n = 1; Activation = Sigmoid } ] }
  let parameters = Map.ofList [(1, { W = W1; b = b1 }); (2, { W = W2; b = b2 }); (3, { W = W3; b = b3 })]

  let AL, caches = _forwardPropagate arch parameters X
  
  caches.Count |> should equal 3
  let expected = [[ 0.03921668; 0.70498921; 0.19734387; 0.04728177]]
  AL |> shouldBeEquivalentM expected

[<Fact>]
let ``Check cost function``() =
  let Y = [[1.0; 1.0; 0.0]] |> toM
  let Ŷ = [[0.8; 0.9; 0.4]] |> toM

  let J = _computeCost Y Ŷ

  J.Should().BeApproximately(0.2797765635793422, precision, System.String.Empty, Array.empty) |> ignore

[<Fact>]
let ``Check linear part of a layer's backward propagation``() =
  let dZ =
    [[ 1.62434536; -0.61175641; -0.52817175; -1.07296862]
     [ 0.86540763; -2.3015387;   1.74481176; -0.7612069 ]
     [ 0.3190391;  -0.24937038;  1.46210794; -2.06014071]] |> toM
  let Aprev =
    [[-0.3224172 ; -0.38405435;  1.13376944; -1.09989127]
     [-0.17242821; -0.87785842;  0.04221375;  0.58281521]
     [-1.10061918;  1.14472371;  0.90159072;  0.50249434]
     [ 0.90085595; -0.68372786; -0.12289023; -0.93576943]
     [-0.26788808;  0.53035547; -0.69166075; -0.39675353]] |> toM
  let W =
    [[-0.6871727 ; -0.84520564; -0.67124613; -0.0126646 ; -1.11731035]
     [ 0.2344157 ;  1.65980218;  0.74204416; -0.19183555; -0.88762896]
     [-0.74715829;  1.6924546 ;  0.05080775; -0.63699565;  0.19091548]] |> toM
  let b =
    [|2.10025514; 0.12015895; 0.61720311|] |> toV

  let (dAprev, dW, db) = _linearBackward dZ { _invalidCache with Aprev = Aprev; W = W; b = b }

  let dAprevExpected =
    [[-1.15171336;  0.06718465; -0.3204696;   2.09812712]
     [ 0.60345879; -3.72508701;  5.81700741; -3.84326836]
     [-0.4319552;  -1.30987417;  1.72354705;  0.05070578]
     [-0.38981415;  0.60811244; -1.25938424;  1.47191593]
     [-2.52214926;  2.67882552; -0.67947465;  1.48119548]]
  let dWExpected =
    [[ 0.07313866; -0.0976715;  -0.87585828;  0.73763362;  0.00785716]
     [ 0.85508818;  0.37530413; -0.59912655;  0.71278189; -0.58931808]
     [ 0.97913304; -0.24376494; -0.08839671;  0.55151192; -0.10290907]]
  let dbExpected =
    [|-0.14713786; -0.11313155; -0.13209101|]

  dAprev |> shouldBeEquivalentM dAprevExpected
  dW |> shouldBeEquivalentM dWExpected
  db |> shouldBeEquivalentV dbExpected

[<Fact>]
let ``Check linear and activation part of a layer's backward propagation``() =
  let dA =
    [[-0.41675785; -0.05626683]] |> toM
  let Aprev =
    [[-2.1361961 ;  1.64027081]
     [-1.79343559; -0.84174737]
     [ 0.50288142; -1.24528809]] |> toM
  let W =
    [[-1.05795222; -0.90900761;  0.55145404]] |> toM
  let b =
    [|2.29220801|] |> toV
  let Z =
    [[ 0.04153939; -1.11792545]] |> toM

  let dAprev, dW, db = _linearActivationBackward dA { Aprev = Aprev; W = W; b = b; Z = Z } ReLU

  let dAprevExpected =
    [[ 0.44090989; -0.0 ]
     [ 0.37883606; -0.0 ]
     [-0.2298228;   0.0 ]]
  let dWExpected =
    [[ 0.44513824;  0.37371418; -0.10478989]]
  let dbExpected =
    [|-0.20837892|]

  dAprev |> shouldBeEquivalentM dAprevExpected
  dW |> shouldBeEquivalentM dWExpected
  db |> shouldBeEquivalentV dbExpected

  let dAprev, dW, db = _linearActivationBackward dA { Aprev = Aprev; W = W; b = b; Z = Z } Sigmoid

  let dAprevExpected =
    [[ 0.11017994;  0.01105339]
     [ 0.09466817;  0.00949723]
     [-0.05743092; -0.00576154]]
  let dWExpected =
    [[ 0.10266786;  0.09778551; -0.01968084]]
  let dbExpected =
    [|-0.05729622|]

  dAprev |> shouldBeEquivalentM dAprevExpected
  dW |> shouldBeEquivalentM dWExpected
  db |> shouldBeEquivalentV dbExpected

[<Fact>]
let ``Check full backward propagation``() =
  let arch =
    { nₓ = 4
      Layers =
        [ { n = 3; Activation = ReLU }
          { n = 1; Activation = Sigmoid } ] }

  let A0 =
    [[ 0.09649747; -1.8634927 ]
     [-0.2773882 ; -0.35475898]
     [-0.08274148; -0.62700068]
     [-0.04381817; -0.47721803]] |> toM
  let W1 =
    [[-1.31386475;  0.88462238;  0.88131804;  1.70957306]
     [ 0.05003364; -0.40467741; -0.54535995; -1.54647732]
     [ 0.98236743; -1.10106763; -1.18504653; -0.2056499 ]] |> toM
  let b1 =
    [| 1.48614836;  0.23671627; -1.02378514|] |> toV       
  let Z1 =
    [[-0.7129932 ;  0.62524497]
     [-0.16051336; -0.76883635]
     [-0.23003072;  0.74505627]] |> toM

  let A1 =
    [[ 1.97611078; -1.24412333]
     [-0.62641691; -0.80376609]
     [-2.41908317; -0.92379202]] |> toM
  let W2 =
    [[-1.02387576;  1.12397796; -0.13191423]] |> toM
  let b2 =
    [|-1.62328545|] |> toV
  let Z2 =
    [[ 0.64667545; -0.35627076]] |> toM

  let caches =
    [(0, _invalidCache)
     (1, { Aprev = A0; W = W1; b = b1; Z = Z1 })
     (2, { Aprev = A1; W = W2; b = b2; Z = Z2 })] |> Map.ofList

  let AL =
    [[1.78862847; 0.43650985]] |> toM
  let Yassess =
    [[1.0; 0.0]] |> toM

  let grads = _backwardPropagate arch AL Yassess caches

  let dA0 = 
    [[ 0.0       ;  0.52257901]
     [ 0.0       ; -0.3269206 ]
     [ 0.0       ; -0.32070404]
     [ 0.0       ; -0.74079187]]
  let dA1 = 
    [[ 0.12913162; -0.44014127]
     [-0.14175655;  0.48317296]
     [ 0.01663708; -0.05670698]]
  let dW1 = 
    [[0.41010002; 0.07807203; 0.13798444; 0.10502167]
     [0.0       ; 0.0       ; 0.0       ; 0.0       ]
     [0.05283652; 0.01005865; 0.01777766; 0.0135308 ]]
  let db1 = 
    [|-0.22007063; 0.0; -0.02835349|]
  let dW2 = 
    [[-0.39202432; -0.13325855; -0.04601089]]
  let db2 = 
    [|0.15187861|]

  grads.[0].dA |> shouldBeEquivalentM dA0
  grads.[1].dA |> shouldBeEquivalentM dA1
  grads.[1].dW |> shouldBeEquivalentM dW1
  grads.[1].db |> shouldBeEquivalentV db1
  grads.[2].dW |> shouldBeEquivalentM dW2
  grads.[2].db |> shouldBeEquivalentV db2

[<Fact>]
let ``Check update parameters``() =
  let arch =
    { nₓ = 2
      Layers =
        [ { n = 3; Activation = ReLU }
          { n = 1; Activation = Sigmoid } ] }
  let W1 = 
    [[-0.41675785; -0.05626683; -2.1361961 ;  1.64027081]
     [-1.79343559; -0.84174737;  0.50288142; -1.24528809]
     [-1.05795222; -0.90900761;  0.55145404;  2.29220801]] |> toM
  let b1 = 
      [| 0.04153939; -1.11792545; 0.53905832 |] |> toV
  let W2 = 
      [[-0.5961597 ; -0.0191305 ;  1.17500122]] |> toM
  let b2 = 
      [|-0.74787095|] |> toV

  let parameters = Map.ofList [(1, { W = W1; b = b1 }); (2, { W = W2; b = b2 })]
    
  let dW1 = 
    [[ 1.78862847;  0.43650985;  0.09649747; -1.8634927 ]
     [-0.2773882 ; -0.35475898; -0.08274148; -0.62700068]
     [-0.04381817; -0.47721803; -1.31386475;  0.88462238]] |> toM
  let db1 = 
    [|0.88131804; 1.70957306; 0.05003364|] |> toV
  let dW2 = 
    [[-0.40467741; -0.54535995; -1.54647732]] |> toM
  let db2 = 
      [|0.98236743|] |> toV

  let grads = [(1, {dA = _invalidMatrix; dW = dW1; db = db1 })
               (2, {dA = _invalidMatrix; dW = dW2; db = db2 })] |> Map.ofList

  let parameters = _updateParameters arch parameters 0.1 grads

  let W1 = 
    [[-0.59562069; -0.09991781; -2.14584584;  1.82662008]
     [-1.76569676; -0.80627147;  0.51115557; -1.18258802]
     [-1.0535704 ; -0.86128581;  0.68284052;  2.20374577]] 
  let b1 = 
    [|-0.04659241; -1.28888275; 0.53405496|]
  let W2 = 
    [[-0.55569196;  0.0354055 ;  1.32964895]] 
  let b2 = 
    [|-0.84610769|]

  parameters.[1].W |> shouldBeEquivalentM W1
  parameters.[1].b |> shouldBeEquivalentV b1
  parameters.[2].W |> shouldBeEquivalentM W2
  parameters.[2].b |> shouldBeEquivalentV b2

