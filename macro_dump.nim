import macros
import sequtils

# static:
#   let stmt: string = """
# XGDMatrixCreateFromFile("/path/to/file/filename", 1, train)
# """
#   let a = parseStmt(stmt)
#   echo a[0][0].repr
#
# XGDMatrixCreateFromFile("/path/to/file/filename", 1, train)

template safe_xgboost(procCall: untyped) =
  let err: int = procCall
  if err != 0:
    let call_stmt: string = callStr           # cast the calling code to string
    let pos = instantiationInfo()             # to get the current file and line number
    # use stderr in Nim to write to stderr, as opposed to fprintf in safe_xgboost
    stderr.writeLine("$1:$2: error in $3: $4\n" % [pos.filename, $pos.line,
      procCall.repr, XGBGetLastError()])
    quit(1)   # replaces C++ exit(1)

safe_xgboost: XGDMatrixCreateFromFile("/path/to/file/filename", 1, train)
# dumptree:
#   let err: int = procCall
#   if err != 0:
#     let call_stmt: string = callStr           # cast the calling code to string
#     let pos = instantiationInfo()             # to get the current file and line number
#     # use stderr in Nim to write to stderr, as opposed to fprintf in safe_xgboost
#     stderr.writeLine("$1:$2: error in $3: $4\n" % [pos.filename, $pos.line,
#       procCall.repr, XGBGetLastError()])
#     quit(1)   # replaces C++ exit(1)

# dumptree:
#   let err: int = XGDMatrixCreateFromFile("/path/to/file/filename", 1, train)
#   if err != 0:
#     let call_stmt: string = "XGDMatrixCreateFromFile"   # cast the calling code to string
#     let pos = instantiationInfo()                       # to get the current file and line number
#     # use stderr in Nim to write to stderr, as opposed to fprintf in safe_xgboost
#     stderr.writeLine("$1:$2: error in $3: $4\n" % [pos.filename, $pos.line,
#       call_stmt, XGBGetLastError()])
#     quit(1)

var
  a: seq[int] = @[1, 2, 3]
  b: seq[int] = @[4, 5, 6]
  c: seq[int] = a & b
echo $foldl(a, a + b)
