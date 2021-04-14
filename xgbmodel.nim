# Load the binary model data at compile time, use it to create and XGBModel,
# and compile into a library

# author: Benjamin Cross
# created: 2021-04-07


import strutils
import json
import xgboost_wrapper


##############################
# compile time calcs
proc getFeatureNames(): seq[cstring] {.compileTime.} =
  ## Load feature names from 'feature_names.txt' at compile time.
  var featureNames: seq[cstring] = newSeq[cstring]()
  for value in staticRead("feature_names.txt").strip().split():
    featureNames.add(cstring(value))
  result = featureNames

# compile time model info and featureNames
const binaryModelStr: string = staticRead("xgb_model_nim.xgb")
const featureNames: seq[cstring] = getFeatureNames()
##############################

# library vars
let
  ptrInitializer: pointer = alloc(0)
  bufferLen: int = binaryModelStr.len()
  # hard-coding missing value for not (same as in other file)
  missing_value: cfloat = 999999999.0
var
  booster: BoosterHandle
  binaryModelData: seq[cuchar] = newSeq[cuchar](bufferLen)
for i, c in pairs(binaryModelStr):
  binaryModelData[i] = c
let
  modelBufferPtr: pointer = cast[pointer](binaryModelData[0].addr())

# parse json payload to features (for use in the modelPredict library function)
proc jsonToFeatures(jsonInputs: cstring):  seq[cfloat] =
    let dataJson: JsonNode = parseJson($jsonInputs)
    var inputSeq: seq[cfloat] = newSeq[cfloat](featureNames.len())
    for i, feature in pairs(featureNames):
      inputSeq[i] = cfloat(dataJson[$feature].getFloat())
    result = inputSeq

########### Beginning of dynamic library definitions
{.push cdecl, exportc, dynlib.}

# load booster model
proc loadModel() =
  booster = ptrInitializer
  safe_xgboost: XGBoosterCreate(nil, 0, booster.unsafeAddr())
  safe_xgboost: XGBoosterLoadModelFromBuffer(booster, modelBufferPtr, bstUlong(bufferLen))

# predict model given json and return json response
proc modelPredict(jsonInputs: cstring, score: ptr cfloat): cstring =
  var
    dmatrix: DMatrixHandle = ptrInitializer
    output_length: bst_ulong
    output_result: ptr cfloat = cast[ptr cfloat](ptrInitializer)
    inputs: seq[cfloat]
  # score model
  try:
    inputs = jsonToFeatures(jsonInputs)
  except:
    return "input json error"

  try:
    safe_xgboost: XGDMatrixCreateFromMat(cast[ptr cfloat](inputs[0].unsafeAddr()),
      1, bst_ulong(inputs.len()), missing_value, dmatrix.addr())
    safe_xgboost: XGBoosterPredict(booster, dmatrix, 0, 0, 0, output_length.addr(),
        output_result.addr())
    safe_xgboost: XGDMatrixFree(dmatrix)
    score[] = output_result[]
  except:
    return "XGBError: $1" % [$XGBGetLastError()]

  return "OK"

# deallocate the booster model
proc freeModel() =
  safe_xgboost: XGBoosterFree(booster)
  booster = ptrInitializer
{.pop.}
########### End of dynamic library definitions

# dealloc(ptrInitializer)
