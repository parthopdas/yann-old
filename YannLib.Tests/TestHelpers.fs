module YannLib.Tests.TestHelpers

open FluentAssertions
open FluentAssertions.Equivalency
open System
open MathNet.Numerics.LinearAlgebra

[<Literal>]
let precision = 1e-7;

let doubleComparisonOptions<'TExpectation> (o: EquivalencyAssertionOptions<'TExpectation>): EquivalencyAssertionOptions<'TExpectation> =
  let action = fun (ctx: IAssertionContext<double>) -> ctx.Subject.Should().BeApproximately(ctx.Expectation, precision, String.Empty, Array.Empty<obj>()) |> ignore
  o.Using<double>(Action<IAssertionContext<double>>(action)).WhenTypeIs<double>()

let shouldBeEquivalentM (a: double list list) (m: Matrix<double>) =
  m.ToArray().Should().BeEquivalentTo(array2D a, doubleComparisonOptions, String.Empty, Array.empty) |> ignore

let shouldBeEquivalentV (a: double array) (v: Vector<double>) =
  v.ToArray().Should().BeEquivalentTo(a, doubleComparisonOptions, String.Empty, Array.empty) |> ignore

let toM (rs: double list list) = rs |> array2D |> CreateMatrix.DenseOfArray

let toV (a: double array) = a |> CreateVector.DenseOfArray
