# Train and XGBoost model using the C API directly
#
# This requires wrapping the functions and types that we will use from the
# XGBoost C API that we need to use, then calling the wrapped functionality
# accordingly to load data and train the model. Several C/C++ calls are also
# replaced by equivalent Nim calls (e.g., exit replaced by quit), and the
# safe_xgboost #define procedure is replaced with ta template that is used
# to block the calls to XGBoost procedures.

# author: Benjamin Cross
# created: 2021-03-31

when defined(macOS):
  let libName: string = "libxgboost.dylib"
elif defined(linux):
  let libName: string = "libxgboost.so"
elif defined(windows):
  let libName: string = "libxgboost.dll"

# use a Nim template to (for the most part) mimic the #define safe_xgboost stmt
# in the XGBoost C API code
template safe_xgboost(procCall: untyped) =
  let err: cint = procCall
  if err != 0:
    let pos = instantiationInfo()   # to get the current file and line number
    # use stderr in Nim to write to stderr, as opposed to fprintf in safe_xgboost
    stderr.writeLine("$1:$2: error in $3: $4\n" % [pos.filename, $pos.line,
      "xgboost api call", $XGBGetLastError()])
    quit(1)   # replaces C++ exit(1)

# C++ wrappers for using the XGBoost C API as in the example here
# https://xgboost.readthedocs.io/en/latest/dev/c__api_8h.html
{.push header: "xgboost/c_api.h", importc, dynlib: libName.}
type
  DMatrixHandle = pointer   # pointer can be used for void * vs. ptr type
  BoosterHandle = pointer
  bst_ulong = culonglong

# API function wrappers
proc XGDMatrixCreateFromFile(fname: cstring, silent: cint, `out`: ptr DMatrixHandle): cint

proc XGDMatrixCreateFromMat(data: ptr cfloat, nrow: bst_ulong, ncol: bst_ulong,
  missing: cfloat, `out`: ptr DMatrixHandle): cint

proc XGBGetLastError(): cstring

proc XGBoosterCreate(dmats: ptr DMatrixHandle, len: bst_ulong, `out`: ptr BoosterHandle): cint

proc XGBoosterSetParam(handle: BoosterHandle, name: cstring, value: cstring): cint

proc XGDMatrixNumCol(handle: DMatrixHandle, `out`: ptr bst_ulong): cint

proc XGDMatrixNumRow(handle: DMatrixHandle, `out`: ptr bst_ulong): cint

proc XGDMatrixSetFloatInfo(handle: DMatrixHandle, field: cstring, array: ptr cfloat, len: bst_ulong): cint

proc XGDMatrixSetDenseInfo(handle: DMatrixHandle, field: cstring, data: pointer,
  size: bst_ulong, `type`: cint): cint

proc XGDMatrixSetUIntInfo(handle: DMatrixHandle, field: cstring, array: ptr cuint, len: bst_ulong): cint

proc XGDMatrixGetFloatInfo(handle: DMatrixHandle, field: cstring, out_len: ptr bst_ulong,
  out_dptr: ptr ptr cfloat): cint

proc XGDMatrixGetUIntInfo(handle: DMatrixHandle, field: cstring, out_len: ptr bst_ulong,
  out_dptr: ptr ptr float): cint

proc XGBoosterUpdateOneIter(handle: BoosterHandle, iter: cint, dtrain: DMatrixHandle): cint

proc XGBoosterEvalOneIter(handle: BoosterHandle, iter: int, dmats: ptr DMatrixHandle,
  evnames: ptr cstring, len: bst_ulong, out_result: ptr cstring): cint

proc XGBoosterPredict(handle: BoosterHandle, dmat: DMatrixHandle, option_mask: cint,
  ntree_limit: cuint, training: cint, out_len: ptr bst_ulong,
  out_result: ptr ptr cfloat): cint

proc XGDMatrixFree(dmatrix: DMatrixHandle): cint

proc XGBoosterFree(booster: BoosterHandle): cint

proc XGBoosterSaveModel(handle: BoosterHandle, fname: cstring): cint

proc XGBoosterLoadModel(handle: BoosterHandle, fname: cstring): cint

proc XGBoosterLoadModelFromBuffer(handle: BoosterHandle, buf: pointer,
  len: bst_ulong): cint
{.pop.}
