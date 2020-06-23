module Yann.DemoApp.Demo2

open YannLib
open YannLib.DNN
open MathNet.Numerics.Data.Matlab

(*
  # DEMO 1

  Basic Transfer Learning: Load previously saved model and resume training

  # NOTES:

  - Ensure the follow are present along with this file
    - https://aimlsutff.blob.core.windows.net/main/deeplearning.ai.C1W4.mat
    - https://aimlsutff.blob.core.windows.net/main/parameters.mat
  
*)

let run callback =
  let dataFile = [() |> Path.getExecutingAssemblyLocation; "deeplearning.ai.C1W4.mat"] |> Path.combine 
  let data = MatlabReader.ReadAll<double>(dataFile, "X", "Y");

  let arch =
    { nₓ = 12288
      Layers =
        [| { n = 20; Activation = ReLU; KeepProb = None }
           { n = 7; Activation = ReLU; KeepProb = None }
           { n = 5; Activation = ReLU; KeepProb = None }
           { n = 1; Activation = Sigmoid; KeepProb = None } |] }

  let hp =
    { Epochs = 2500
      α = 0.0075
      HeScale = 1.
      λ = None
      Optimization = NoOptimization
      BatchSize = BatchSizeAll }

  let ps0 = DataLoaders.loadParameters 4 "parameters.mat"
  let parameters = trainNetwork_ 1 (Some ps0) callback arch hp data.["X"] data.["Y"]
  // Save parameters if necessary
  parameters |> ignore
