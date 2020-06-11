open Yann.DemoApp

let callback epoch elapsed cost accuracy =
  if epoch % 100 = 0 then
    printfn "[%O] Epoch=%04d Cost=%2.8f Accuracy=%2.2f" elapsed epoch cost accuracy
  else
    ()

[<EntryPoint>]
let main _ =
  MathNet.Numerics.Control.UseNativeMKL();

  Demo1.run callback

  0
