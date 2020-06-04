open MathNet.Numerics.Data.Matlab
open YannLib.Core

let callback epoch elapsed cost accuracy =
  if epoch % 100 = 0 then
    printfn "[%O] Epoch=%d Cost=%f Accuracy=%f" elapsed epoch cost accuracy
  else
    ()

let loadParameters L (matFile: string): Parameters = 
  let paramNames =
    [ for pName in ["W"; "b"] do
      for l in 1 .. L -> sprintf "%s%d" pName l ] |> List.toArray
  let rawParameters = MatlabReader.ReadAll<double>(matFile, paramNames);
  let _folder acc e =
    acc
    |> Map.add e { W = rawParameters.[sprintf "W%d" e]; b = rawParameters.[sprintf "b%d" e].EnumerateColumns() |> Seq.exactlyOne }

  [ 1 .. L ] |> List.fold _folder Map.empty

[<EntryPoint>]
let main _ =
  MathNet.Numerics.Control.UseNativeMKL();

  let data = MatlabReader.ReadAll<double>("deeplearning.ai.C1W4.mat", "X", "Y");

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

  let ps0 = loadParameters 4 "parameters.mat"
  let parameters = trainNetwork (Parameters ps0) callback arch data.["X"] data.["Y"] hp

  0
