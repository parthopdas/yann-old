module YannLib.Tests.PerfTests

open global.Xunit
open FsUnit.Xunit
open MathNet.Numerics.LinearAlgebra
open System.Diagnostics

[<Fact>]
let ``Check MLK install``() =
  MathNet.Numerics.Control.UseNativeMKL();
  let m1 = DenseMatrix.create 1000 1000 1.0

  let sw = Stopwatch()
  let mutable totalMS = 0L
  let count = 100

  for _ in 1 .. count do
    sw.Start()
    let _ = m1.Multiply(m1)
    sw.Stop()
    totalMS <- totalMS + sw.ElapsedMilliseconds

  totalMS |> should be (lessThan 100000L)
