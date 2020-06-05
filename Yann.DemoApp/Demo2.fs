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
        [ { n = 20; Activation = ReLU }
          { n = 7; Activation = ReLU }
          { n = 5; Activation = ReLU }
          { n = 1; Activation = Sigmoid } ] }

  let hp =
    { Epochs = 2500
      α = 0.0075 }

  let ps0 = DataLoaders.loadParameters 4 "parameters.mat"
  let parameters = trainNetwork (Parameters ps0) callback arch data.["X"] data.["Y"] hp
  // Save parameters if necessary
  parameters |> ignore
