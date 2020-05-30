module YannLib.Tests.TestHelpers

open FluentAssertions
open FluentAssertions.Equivalency
open System

[<Literal>]
let comparisonPrecision = 1e-6;

let doubleComparisonOptions<'TExpectation> (o: EquivalencyAssertionOptions<'TExpectation>): EquivalencyAssertionOptions<'TExpectation> =
  let action = fun (ctx: IAssertionContext<double>) -> ctx.Subject.Should().BeApproximately(ctx.Expectation, comparisonPrecision, String.Empty, Array.Empty<obj>()) |> ignore
  o.Using<double>(Action<IAssertionContext<double>>(action)).WhenTypeIs<double>()
