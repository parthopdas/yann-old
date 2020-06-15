module Yann.DemoApp.Demo1

open YannLib.DNN
open MathNet.Numerics.Data.Matlab

(*
  # DEMO 1

  Cat vs non-Cats classification exercise from DL.AI: Course 1 / Week 4 

  # NOTES:

  - Ensure the follow are present along with this file
    - https://aimlsutff.blob.core.windows.net/main/deeplearning.ai.C1W4.mat
  
*)

let run callback =
  let dataFile = [() |> Path.getExecutingAssemblyLocation; "deeplearning.ai.C1W4.mat"] |> Path.combine 
  let data = MatlabReader.ReadAll<double>(dataFile, "X", "Y");

  let arch =
    { nₓ = 12288
      Layers =
        [| { n = 20; Activation = ReLU; KeepProb = Some 0.9 }
           { n = 7; Activation = ReLU; KeepProb = Some 1. }
           { n = 5; Activation = ReLU; KeepProb = Some 1. }
           { n = 1; Activation = Sigmoid; KeepProb = None } |] }

  let hp =
    { Epochs = 2500
      α = 0.0075
      HeScale = 1.
      λ = Some 0.1
      BatchSize = BatchSizeAll }

  trainNetwork (Seed 1) callback arch data.["X"] data.["Y"] hp |> ignore


