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

# defining convenience type for CArray
type
  CArray[T] = UncheckedArray[T]

# C++ wrappers for using the XGBoost C API as in the example here
# https://xgboost.readthedocs.io/en/latest/dev/c__api_8h.html
{.push header: "xgboost/c_api.h", importc, dynlib: libName.}
type
  DMatrixHandle = pointer   # pointer can be used for void * vs. ptr type
  BoosterHandle = pointer
  # for CArrays that do not have a specified size for readability purposes
  # some trickiness though. Arrays in Nims are NEVER pointers. So the types passed
  # below for a C++ arg likeso, `DMatrixHandle dmats[]`, will be `ptr CArray[DMatrixHandle]`
  bst_ulong = culonglong

proc XGDMatrixCreateFromFile(fname: cstring, silent: cint, `out`: DMatrixHandle): cint
# note the backticks below for the `out` parameter mask the 'out' keyword (this may
# not be necessary, could just use a diff arg name?)
proc XGDMatrixCreateFromMat(data: ptr cfloat, nrow: bst_ulong, ncol: bst_ulong,
  missing: cfloat, `out`: ptr DMatrixHandle): cint
proc XGBGetLastError(): cstring
proc XGBoosterCreate (dmats: ptr CArray[DMatrixHandle], len: bst_ulong, `out`: ptr BoosterHandle): cint
proc XGBoosterSetParam(handle: BoosterHandle, name: cstring, value: cstring): cint
proc XGDMatrixSetFloatInfo (handle: DMatrixHandle, field: cstring, array: ptr cfloat, len: bst_ulong): cint
proc XGBoosterUpdateOneIter(handle: BoosterHandle, iter: cint, dtrain: DMatrixHandle): cint
# below, as per Nim docs cstringArray is equivalent to ptr UncheckedArray[cstring] = ptr CArray[cstring]
proc XGBoosterEvalOneIter(handle: BoosterHandle, iter: int, dmats: ptr CArray[DMatrixHandle],
  evnames: cstringArray, len: bst_ulong, out_result: cstringArray): cint
proc XGBoosterPredict (handle: BoosterHandle, dmat: DMatrixHandle, option_mask: cint,
  ntree_limit: cuint, training: cint, out_len: ptr bst_ulong,
  out_result: ptr CArray[ptr cfloat]): cint
{.pop.}
