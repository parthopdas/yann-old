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

  let _folder (ttlMs, m: Matrix<double>) _ =
    sw.Start()
    let m = m.Multiply(m)
    sw.Stop()
    ttlMs + sw.ElapsedMilliseconds, m
    
  let totalMS, _ =
    seq { 1 .. 100 }
    |> Seq.fold _folder (0L, m1)

  totalMS |> should be (lessThan 120000L)
