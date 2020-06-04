namespace YannLib

open MathNet.Numerics.LinearAlgebra
open System.Runtime.CompilerServices
open System.Linq

[<Extension>]
type MathNetExtensions() =
  [<Extension>]
  static member inline Shape(m: Matrix<double>) = [|m.RowCount; m.ColumnCount|]
  [<Extension>]
  static member inline ShapeString(m: Matrix<double>) = sprintf "%dx%d" m.RowCount m.ColumnCount

  [<Extension>]
  static member inline Shape(m: Vector<double>) = [|m.Count; 1|]
  [<Extension>]
  static member inline ShapeString(m: Vector<double>) = sprintf "%dx1" m.Count

  [<Extension>]
  static member inline BroadcastC(v: Vector<double>, count: int) =
    Matrix<double>.Build.DenseOfColumnVectors(Enumerable.Repeat(v, count))
