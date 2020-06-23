open Yann.DemoApp
open System
open System.Collections.Generic

let callback (Js: List<double>) (accs: List<double>) epoch (elapsed: TimeSpan) cost accuracy =
  Js.Add cost
  accs.Add accuracy

  if epoch % 100 = 0 then
    printfn "[%04.6f] Epoch=%04d Cost=%2.8f Accuracy=%2.2f" elapsed.TotalSeconds epoch cost accuracy
  else
    ()

[<EntryPoint>]
let main _ =
  MathNet.Numerics.Control.UseNativeMKL();

  let Js = List<double>(2500)
  let accs = List<double>(2500)
  Demo1.run (callback Js accs)

  Helpers.plot "Demo1" Js accs |> Helpers.openFileWithShell |> ignore

  0
