module Yann.DemoApp.Helpers

open System.Diagnostics
open System.Collections.Generic
open System
open System.Drawing

let openFileWithShell file =
  let psi = ProcessStartInfo file
  psi.UseShellExecute <- true

  Process.Start psi

// NOTE: From https://swharden.com/scottplot/cookbooks/4.0.36/#plottypes-point-plot-points
let plot name (Js: IReadOnlyList<double>) (accs: IReadOnlyList<double>) =
  let plt = ScottPlot.Plot(600, 400);

  let x = seq { for i = 0 to Js.Count - 1 do i } |> Seq.map double |> Seq.toArray
  let J = Js |> Seq.toArray
  let acc = accs |> Seq.toArray

  plt.PlotScatter(x, J, Nullable(Color.Blue), 0.1, 0.1, "Cost") |> ignore
  plt.PlotScatter(x, acc, Nullable(Color.Orange), 0.1, 0.1, "Accuracy") |> ignore

  plt.Title name
  plt.XLabel "Epoch"

  let fileName = "CostFunction.png"
  plt.SaveFig fileName

  fileName
