open MathNet.Numerics.Data.Matlab
open YannLib.Core

let callback epoch elapsed cost accuracy =
  if epoch % 100 = 0 then
    printfn "[%O] Epoch=%d Cost=%f Accuracy=%f" elapsed epoch cost accuracy
  else
    ()

[<EntryPoint>]
let main _ =
  MathNet.Numerics.Control.UseNativeMKL();

  let data = MatlabReader.ReadAll<double>("dl.al.cats.mat", "X", "Y");

  let arch =
    { nₓ = 12288
      Layers =
        [ { n = 20; Activation = ReLU }
          { n = 7; Activation = ReLU }
          { n = 5; Activation = ReLU }
          { n = 1; Activation = Sigmoid } ] }

  let hp =
    { Epochs = 2500
      α = 0.001 }
  let parameters = trainNetwork 1 callback arch data.["X"] data.["Y"] hp
  0
