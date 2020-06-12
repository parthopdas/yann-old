open Yann.DemoApp
open System

let callback epoch (elapsed: TimeSpan) cost accuracy =
  if epoch % 100 = 0 then
    printfn "[%04.6f] Epoch=%04d Cost=%2.8f Accuracy=%2.2f" elapsed.TotalSeconds epoch cost accuracy
  else
    ()

[<EntryPoint>]
let main _ =
  MathNet.Numerics.Control.UseNativeMKL();

  Demo1.run callback

  0
