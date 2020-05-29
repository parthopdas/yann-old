[<AutoOpen>]
module YaFunTK

[<RequireQualifiedAccess>]
module Debug = 
    open System.Diagnostics
    
    let inline dprintfn fmt = Printf.ksprintf Debug.WriteLine fmt


[<RequireQualifiedAccess>]
module Prelude = 
    open System
    open System.Collections.Generic
    open System.Collections.ObjectModel
    
    let inline (==) a b = obj.ReferenceEquals(a, b)
    let inline (===) a b = LanguagePrimitives.PhysicalEquality a b    
    let inline toStr x = x.ToString()
    let inline ct x = fun _ -> x
    let inline flip f a b = f b a
    let inline flip3 f a b c = f c a b
    let inline curry f a b = f (a, b)
    let inline uncurry f (a, b) = f a b
    let inline curry3 f a b c = f (a, b, c)
    let inline uncurry3 f (a, b, c) = f a b c
    let inline tuple2 a b = a, b
    let inline tuple3 a b c = a, b, c
    let inline swap (a, b) = (b, a)
    let undefined<'T> : 'T = raise (NotImplementedException("result was implemented as undefined"))
    
    let inline tee fn x = 
        fn x |> ignore
        x
    
    let roDict x = x |> dict |> ReadOnlyDictionary<_, _> :> IReadOnlyDictionary<_, _>


[<RequireQualifiedAccess>]
module Path = 
    open System
    open System.IO
    open System.Reflection
    
    let combine = List.reduce (Prelude.curry Path.Combine)

    let inline getAssemblyPath (a: Assembly) = 
        a.CodeBase
        |> Uri
        |> fun x -> x.LocalPath
        |> Path.GetFullPath
    
    let getExecutingAssemblyLocation = Assembly.GetExecutingAssembly >> getAssemblyPath >> Path.GetDirectoryName

[<RequireQualifiedAccess>]
[<CompilationRepresentation(CompilationRepresentationFlags.ModuleSuffix)>]
module Environment = 
    open System

    let IsMono = "Mono.Runtime" |> Type.GetType |> isNull |> not


[<RequireQualifiedAccess>]
[<CompilationRepresentation(CompilationRepresentationFlags.ModuleSuffix)>]
module Serialization = 
    open System.Reflection
    open Microsoft.FSharp.Reflection

    let knownTypes<'T> () =
        typeof<'T>.GetNestedTypes(BindingFlags.Public ||| BindingFlags.NonPublic) 
        |> Array.filter FSharpType.IsUnion


[<RequireQualifiedAccess>]
[<CompilationRepresentation(CompilationRepresentationFlags.ModuleSuffix)>]
module String = 
    open Microsoft.FSharp.Reflection
    open System
    
    let toUnionCase<'a> s = 
        match FSharpType.GetUnionCases typeof<'a> |> Array.filter (fun case -> case.Name = s) with
        | [| case |] -> Some(FSharpValue.MakeUnion(case, [||]) :?> 'a)
        | _ -> None

    /// Also, invariant culture
    let equals (a : string) (b : string) =
        a.Equals(b, StringComparison.InvariantCulture)

    /// Also, invariant culture
    let equalsCaseInsensitive (a : string) (b : string) =
        a.Equals(b, StringComparison.InvariantCultureIgnoreCase)
    
    /// Compare ordinally with ignore case.
    let equalsOrdinalCI (str1 : string) (str2 : string) =
        String.Equals(str1, str2, StringComparison.OrdinalIgnoreCase)

    /// Ordinally compare two strings in constant time, bounded by the length of the
    /// longest string.
    let equalsConstantTime (str1 : string) (str2 : string) = 
        let mutable xx = uint32 str1.Length ^^^ uint32 str2.Length
        let mutable i = 0
        while i < str1.Length && i < str2.Length do
            xx <- xx ||| uint32 (int str1.[i] ^^^ int str2.[i])
            i <- i + 1
        xx = 0u

    let toLowerInvariant (str : string) =
        str.ToLowerInvariant()

    let replace (find : string) (replacement : string) (str : string) =
        str.Replace(find, replacement)

    let isEmpty (s : string) =
        s.Length = 0

    let trim (s : string) =
        s.Trim()
  
    let trimc (toTrim : char) (s : string) =
        s.Trim toTrim
  
    let trimStart (s : string) =
        s.TrimStart()
  
    let split (c : char) (s : string) =
        s.Split c |> Array.toList
  
    let splita (c : char) (s : string) =
        s.Split c
  
    let startsWith (substring : string) (s : string) =
        s.StartsWith substring
  
    let contains (substring : string) (s : string) =
        s.Contains substring
  
    let substring index (s : string) =
        s.Substring index
  
    let toCharArray (s : string) =
        s.ToCharArray()
  
    let fromCharArray (cs : char []) =
        String cs


[<RequireQualifiedAccess>]
[<CompilationRepresentation(CompilationRepresentationFlags.ModuleSuffix)>]
module StringBuilder = 
    open System.Text

    let appendLine (sb: StringBuilder) x =
        sb.AppendLine(x)


module Result =
    let returnM = Ok

    let inline map f = function
        | Ok v -> v |> f |> Ok
        | Error v -> Error v
    let (<!>) = map
    let inline (|>>) v f = map f v

    let inline apply f v =
        match f, v with
        | Ok f, Ok v -> Ok(f v)
        | Error f, _ -> Error f
        | _, Error v -> Error v
    let inline (<*>) f v = apply f v

    let inline bind f =
        function
        | Ok v' -> f v'
        | Error v' -> Error v'
    let inline (>>=) v f = bind f v
    let inline (=<<) v f = bind f v
    let inline (>=>) f g = fun x -> f x >>= g
    let inline (<=<) x = Prelude.flip (>=>) x

    let inline ( *>* ) r1 r2 = Prelude.tuple2 <!> r1 <*> r2

    let inline ( *> ) r1 r2 = snd <!> r1 <*> r2
    let inline ( >>. ) r1 r2 = snd <!> r1 <*> r2

    let inline ( <* ) r1 r2 = fst <!> r1 <*> r2

    let inline fold fS fF = 
        function
        | Ok v -> fS v
        | Error v -> fF v

[<RequireQualifiedAccess>]
[<CompilationRepresentation(CompilationRepresentationFlags.ModuleSuffix)>]
module Option =
    open System

    let inline ofNull value =
        if obj.ReferenceEquals(value, null) then None else Some value

    let inline ofNullable (value: Nullable<'T>) =
        if value.HasValue then Some value.Value else None

    let inline toNullable (value: 'T option) =
        match value with
        | Some x -> Nullable<_> x
        | None -> Nullable<_> ()

    let inline attempt (f: unit -> 'T) = try Some <| f() with _ -> None

    /// Gets the value associated with the option or the supplied default value.
    let inline getOrElse v =
        function
        | Some x -> x
        | None -> v

    /// Gets the option if Some x, otherwise the supplied default value.
    let inline orElse v =
        function
        | Some x -> Some x
        | None -> v

    /// Gets the value if Some x, otherwise try to get another value by calling a function
    let inline getOrTry f =
        function
        | Some x -> x
        | None -> f()

    /// Gets the option if Some x, otherwise try to get another value
    let inline orTry f =
        function
        | Some x -> Some x
        | None -> f()

    /// Some(Some x) -> Some x | None -> None
    let inline flatten x =
        match x with
        | Some x -> x
        | None -> None

    let inline toList x =
        match x with
        | Some x -> [x]
        | None -> []

    let inline iterElse someAction noneAction opt =
        match opt with
        | Some x -> someAction x
        | None   -> noneAction ()


[<RequireQualifiedAccess>]
[<CompilationRepresentation(CompilationRepresentationFlags.ModuleSuffix)>]
module Async = 
    open System

    let result = async.Return
    let map f value = async { let! v = value
                              return f v }
    let bind f xAsync = async { let! x = xAsync
                                return! f x }
    
    let withTimeout timeoutMillis operation = 
        async { 
            let! child = Async.StartChild(operation, timeoutMillis)
            try 
                let! result = child
                return Some result
            with :? TimeoutException -> return None
        }
    
    let apply fAsync xAsync = async { 
                                  // start the two asyncs in parallel
                                  let! fChild = Async.StartChild fAsync
                                  let! xChild = Async.StartChild xAsync
                                  // wait for the results
                                  let! f = fChild
                                  let! x = xChild
                                  // apply the function to the results
                                  return f x }
    let lift2 f x y = apply (apply (result f) x) y
    let lift3 f x y z = apply (apply (apply (result f) x) y) z
    let lift4 f x y z a = apply (apply (apply (apply (result f) x) y) z) a
    let lift5 f x y z a b = apply (apply (apply (apply (apply (result f) x) y) z) a) b
    
    module Operators = 
        let inline (>>=) m f = bind f m
        let inline (=<<) f m = bind f m
        let inline (<*>) f m = apply f m
        let inline (<!>) f m = map f m
        let inline ( *> ) m1 m2 = lift2 (fun _ x -> x) m1 m2
        let inline (<*) m1 m2 = lift2 (fun x _ -> x) m1 m2


[<RequireQualifiedAccess>]
module App = 
    open System.Reflection
    
    /// Gets the calling assembly's informational version number as a string
    let getVersion() = 
        Assembly.GetCallingAssembly().GetCustomAttribute<AssemblyInformationalVersionAttribute>().InformationalVersion

[<RequireQualifiedAccess>]
module UTF8 = 
    open System.Text
    open System
    
    type Base64String = string

    let private utf8 = Encoding.UTF8
    
    /// Convert the full buffer `b` filled with UTF8-encoded strings into a CLR
    /// string.
    let toString (bs : byte []) = utf8.GetString bs
    
    /// Convert the byte array to a string, by indexing into the passed buffer `b`
    /// and taking `count` bytes from it.
    let toStringAtOffset (b : byte []) (index : int) (count : int) = utf8.GetString(b, index, count)
    
    /// Get the UTF8-encoding of the string.
    let bytes (s : string) = utf8.GetBytes s
    
    /// Convert the passed string `s` to UTF8 and then encode the buffer with
    /// base64.
    let encodeBase64 : string -> Base64String = bytes >> Convert.ToBase64String
    
    /// Convert the passed string `s`, assumed to be a valid Base64 encoding, to a
    /// CLR string, going through UTF8.
    let decodeBase64 : Base64String -> string = Convert.FromBase64String >> toString