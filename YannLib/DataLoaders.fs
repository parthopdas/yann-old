module YannLib.DataLoaders

open YannLib.DNN
open MathNet.Numerics.Data.Matlab

let loadParameters L (matFileName: string): Parameters = 
  let matFile = [() |> Path.getExecutingAssemblyLocation; matFileName] |> Path.combine 
  let paramNames =
    [ for pName in ["W"; "b"] do
      for l in 1 .. L -> sprintf "%s%d" pName l ] |> List.toArray
  let rawParameters = MatlabReader.ReadAll<double>(matFile, paramNames);
  let _folder acc e =
    acc
    |> Map.add e { W = rawParameters.[sprintf "W%d" e]; b = rawParameters.[sprintf "b%d" e].EnumerateColumns() |> Seq.exactlyOne }

  [ 1 .. L ] |> List.fold _folder Map.empty
