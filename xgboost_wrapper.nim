## Train an XGBoost model using the C API directly
##
## This requires wrapping the functions and types that we will use from the
## XGBoost C API, then calling the wrapped functionality accordingly to load data
## and train the model. The safe_xgboost #define procedure is replaced with a
## template that is used to wrap the calls to XGBoost procedures.

# author: Benjamin Cross
# created: 2021-03-31


when defined(macos) or defined(macosx):
  const libName: string = "libxgboost.dylib"
elif defined(linux):
  const libName: string = "libxgboost.so"
elif defined(windows):
  const libName: string = "libxgboost.dll"

# Wrappers for using the XGBoost C API as in the example here
# https://xgboost.readthedocs.io/en/latest/dev/c__api_8h.html
{.push cdecl, header: "xgboost/c_api.h", importc, dynlib: libName.}
type
  DMatrixHandle* = pointer   # pointer can be used for void * vs. ptr type
  BoosterHandle* = pointer
  bst_ulong* = culonglong

# API function wrappers
proc XGDMatrixCreateFromFile*(fname: cstring, silent: cint, `out`: ptr DMatrixHandle): cint

proc XGDMatrixCreateFromMat*(data: ptr cfloat, nrow: bst_ulong, ncol: bst_ulong,
  missing: cfloat, `out`: ptr DMatrixHandle): cint

proc XGDMatrixCreateFromMat_omp*(data: ptr cfloat, nrow: bst_ulong, ncol: bst_ulong,
  missing: cfloat, `out`: ptr DMatrixHandle, thread: cint): cint

proc XGBGetLastError*(): cstring

proc XGBoosterCreate*(dmats: ptr DMatrixHandle, len: bst_ulong, `out`: ptr BoosterHandle): cint

proc XGBoosterSetParam*(handle: BoosterHandle, name: cstring, value: cstring): cint

proc XGDMatrixNumCol*(handle: DMatrixHandle, `out`: ptr bst_ulong): cint

proc XGDMatrixNumRow*(handle: DMatrixHandle, `out`: ptr bst_ulong): cint

proc XGDMatrixSetFloatInfo*(handle: DMatrixHandle, field: cstring, array: ptr cfloat, len: bst_ulong): cint

proc XGDMatrixSetDenseInfo*(handle: DMatrixHandle, field: cstring, data: pointer,
  size: bst_ulong, `type`: cint): cint

proc XGDMatrixSetUIntInfo*(handle: DMatrixHandle, field: cstring, array: ptr cuint, len: bst_ulong): cint

proc XGDMatrixGetFloatInfo*(handle: DMatrixHandle, field: cstring, out_len: ptr bst_ulong,
  out_dptr: ptr ptr cfloat): cint

proc XGDMatrixGetUIntInfo*(handle: DMatrixHandle, field: cstring, out_len: ptr bst_ulong,
  out_dptr: ptr ptr float): cint

proc XGDMatrixSetStrFeatureInfo*(handle: DMatrixHandle, field: cstring,
  features: ptr cstring, size: bst_ulong): cint

proc XGDMatrixGetStrFeatureInfo*(handle: DMatrixHandle, field: cstring,
  size: ptr bst_ulong, out_features: ptr ptr cstring): cint

proc XGBoosterUpdateOneIter*(handle: BoosterHandle, iter: cint, dtrain: DMatrixHandle): cint

proc XGBoosterEvalOneIter*(handle: BoosterHandle, iter: int, dmats: ptr DMatrixHandle,
  evnames: ptr cstring, len: bst_ulong, out_result: ptr cstring): cint

proc XGBoosterPredict*(handle: BoosterHandle, dmat: DMatrixHandle, option_mask: cint,
  ntree_limit: cuint, training: cint, out_len: ptr bst_ulong,
  out_result: ptr ptr cfloat): cint

proc XGDMatrixFree*(dmatrix: DMatrixHandle): cint

proc XGBoosterFree*(booster: BoosterHandle): cint

proc XGBoosterSaveModel*(handle: BoosterHandle, fname: cstring): cint

proc XGBoosterLoadModel*(handle: BoosterHandle, fname: cstring): cint

proc XGBoosterLoadModelFromBuffer*(handle: BoosterHandle, buf: pointer,
  len: bst_ulong): cint
{.pop.}


# create error type for XGB to be raised in safe_xgboost template
type
  XGBError* = object of CatchableError

# use a Nim template to (for the most part) mimic the #define safe_xgboost stmt
# in the XGBoost C API code
template safe_xgboost*(procCall: untyped) =
  ## Mimic the safe_xgboost #define procedure in the C API
  ##
  ## Usage for some arbitrary call, XGBFunctionCall(...)
  ##   safe_xgboost: XGBFunctionCall(...)
  let err: cint = procCall
  if err != 0:
    let
      pos = instantiationInfo()   # to get the current file and line number
      errMsg: string = $XGBGetLastError()
    # use stderr in Nim to write to stderr, as opposed to fprintf in safe_xgboost
    stderr.writeLine("$1:$2: error in $3: $4\n" % [pos.filename, $pos.line,
      "xgboost api call", errMsg])
    # raise exception
    raise newException(XGBError, errMsg)
    # quit(1)   # replaces C++ exit(1)
