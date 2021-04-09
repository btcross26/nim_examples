# compiler options to include - note --clib option does not work correctly
# so need to use --passL instead to specify xgboost library. Also, note the use
# of anaconda here since this is being run in a conda environment with nim
# installed.
# --cincludes:$HOME/anaconda3/envs/nim_demo/include
# --clibdir:$HOME/anaconda3/envs/nim_demo/lib
# --passL:"-lxgboost"
# --passL:"-rpath $HOME/anaconda3/envs/nim_demo/lib"
# --threads-on

# author: Benjamin Cross
# created: 2021-03-31

import strutils
import strformat
from re import nil   # use qualified name here
import nimpy, nimpy/[py_types, raw_buffers]
import xgboost_wrapper


##### UNSAFE #####
# create a dummy allocated space for ptr initialization when a
# function requires a pointer argument for the purpose of copying data.
# c demo is tricky here - to work in Nim, must initialize the void pointer as
# it seems that the default intialization in Nim is nil. This pointer is used
# as a common allocated space to intialize pointers so that they are not nil
# and do not throw errors in the wrapped XGBoost interfaces.
let ptrInitializer: pointer = alloc(0)

# the code below is taken directly from the xgboost C API example and modified
# accordingly to work with Nim syntax and the example dataset
# https://xgboost.readthedocs.io/en/latest/tutorials/c_api_tutorial.html

# Skip step 1 below, since step 2 will be used.
# 1. If the dataset is available in a file, it can be loaded into a DMatrix
#    object using the XGDMatrixCreateFromFile
# 2. You can also create a DMatrix object from a 2D Matrix using the
#    XGDMatrixCreateFromMat function

# let's use Python here to create the data as per nimpy instructions
# it's actually not THAT different from Python, but not all convenience is available
# since everything returned is a PyObject. For example, I can't unpack a PyObject
# tuple directly in Nim without creating a Nim tuple, as shown below.
let
  sklearn_datasets = pyImport("sklearn.datasets")
  sklearn_ms = pyImport("sklearn.model_selection")
  datasets = sklearn_datasets.make_classification(n_samples=10000, n_features=10,
    n_informative=5, n_redundant=2, random_state=17)
  (X, y) = (datasets[0], datasets[1])
  splits = sklearn_ms.train_test_split(X, y, test_size=0.2, random_state=17, stratify=y)
var
  (X_train, X_test, y_train, y_test) = (splits[0], splits[1], splits[2], splits[3])

# Cast the above arrays to float32 for use with xgboost. Note: Have to use the
# string type shortcuts here to avoid collisions with Nim. For example,np.float32
# resolves in Nim to float32(np). So Nim thinks you are trying to cast np, which
# is a PyObject, to the native float32 type, and the compiler will tell you so.
X_train = X_train.astype("f4")
X_test = X_test.astype("f4")
y_train = y_train.astype("f4")
y_test = y_test.astype("f4")

# Get sizes. Use the nimpy `to` proc to cast from a PyObject to the type I want.
# The `to` proc basically dereferences the PyObject pointer and skips over the
# initial memory overhead required of the Python object for storing reference
# counts etc., all the while casting the pointer back to a Nim type and
# dereferencing via automatic dereferencing . The definition from nimpy:
# proc to*(v: PyObject, T: typedesc): T {.inline.} =
#   when T is void:
#     discard
#   else:
#     pyObjToNim(v.rawPyObj, result)
#
# There is also an overloaded version of this that works with type PPyObject,
# which is the type of PyObject.rawPyObj
let
  train_rows: bst_ulong = X_train.shape[0].to(bst_ulong)
  train_cols: bst_ulong = X_train.shape[1].to(bst_ulong)
  test_rows: bst_ulong = X_test.shape[0].to(bst_ulong)
  test_cols: bst_ulong = X_test.shape[1].to(bst_ulong)

# Create buffers to the numpy data created above using nimpy functionality
var
  X_train_buffer: RawPyBuffer
  X_test_buffer: RawPyBuffer
  y_train_buffer: RawPyBuffer
  y_test_buffer: RawPyBuffer
getBuffer(X_train, X_train_buffer, PyBUF_STRIDES)
getBuffer(X_test, X_test_buffer, PyBUF_STRIDES)
getBuffer(y_train, y_train_buffer, PyBUF_STRIDES)
getBuffer(y_test, y_test_buffer, PyBUF_STRIDES)

let
  # hard-coding this for now
  missing_value: cfloat = 999999999.0
var
  train: DMatrixHandle = ptrInitializer
  test: DMatrixHandle = ptrInitializer

# serialized version
safe_xgboost: XGDMatrixCreateFromMat(cast[ptr cfloat](X_train_buffer.buf),
  train_rows, train_cols, missing_value, train.unsafeAddr())
safe_xgboost: XGDMatrixCreateFromMat(cast[ptr cfloat](X_test_buffer.buf),
  test_rows, test_cols, missing_value, test.unsafeAddr())

# # parallelized version
# safe_xgboost: XGDMatrixCreateFromMat_omp(cast[ptr cfloat](X_train_buffer.buf),
#   train_rows, train_cols, 999999999.0, train.unsafeAddr(), -1)
# safe_xgboost: XGDMatrixCreateFromMat_omp(cast[ptr cfloat](X_test_buffer.buf),
#   test_rows, test_cols, 999999999.0, test.unsafeAddr(), -1)

# set feature names
var feature_names: seq[cstring] = newSeq[cstring](10)
for i in 0..<10:
  feature_names[i] = "f$1" % [$i]
safe_xgboost: XGDMatrixSetStrFeatureInfo(train, "feature_name", feature_names[0].addr(),
  10)

# 3. Create a Booster object for training & testing on dataset using XGBoosterCreate
var booster: BoosterHandle = ptrInitializer

# need this at compiletime to create array below (could have used {compileTime} pragma also)
const eval_dmats_size: bst_ulong = 2

# Below was tricky, but arrays are never pointers in Nim, so to get the interface
# right and get used to this will take some work
let
  train_test_array: array[eval_dmats_size, DMatrixHandle] = [train, test]
  eval_dmats: ptr DMatrixHandle = cast[ptr DMatrixHandle](train_test_array.unsafeAddr())
safe_xgboost: XGBoosterCreate(eval_dmats, eval_dmats_size, booster.unsafeAddr())


# 4. For each DMatrix object, set the labels using XGDMatrixSetFloatInfo. Later
# you can access the label using XGDMatrixGetFloatInfo (see example link)
safe_xgboost: XGDMatrixSetFloatInfo(train, "label", cast[ptr cfloat](y_train_buffer.buf), train_rows)
safe_xgboost: XGDMatrixSetFloatInfo(test, "label", cast[ptr cfloat](y_test_buffer.buf), test_rows)

# note: UncheckedArray is unintuitive when it comes to passing a pointer to a
# C function for purposes of storing data. It seems easier and more fool-proof to
# stick to using pointers and pointer arithmetic.
var
  out_len: bst_ulong
  out_dptr: ptr ptr cfloat = cast[ptr ptr cfloat](ptrInitializer)
safe_xgboost: XGDMatrixGetFloatInfo(train, "label", out_len.addr(), out_dptr)

echo "Number of labels in training set: $1" % [$out_len]
for i in 0..<out_len:
  var value: cfloat = cast[ptr cfloat](cast[uint](out_dptr[]) + uint(sizeof(cfloat)) * i)[]
  echo "Label $1: $2" % [$i, &"{value:.0f}"]
  if i >= 10:
    break
echo()


# 5. Set the parameters for the Booster object according to the requirement
# using XGBoosterSetParam . Check out the full list of parameters available
# here.
safe_xgboost: XGBoosterSetParam(booster, "booster", "gblinear")
safe_xgboost: XGBoosterSetParam(booster, "objective", "binary:logistic")
safe_xgboost: XGBoosterSetParam(booster, "eval_metric", "auc")
safe_xgboost: XGBoosterSetParam(booster, "max_depth", "3")
safe_xgboost: XGBoosterSetParam(booster, "eta", "0.05")   # default eta  = 0.3


# 6. Train & evaluate the model using XGBoosterUpdateOneIter and
# XGBoosterEvalOneIter respectively.
let
  num_of_iterations: cint = 1000
  early_stopping_rounds: int = 25
  check_index: int = early_stopping_rounds + 1
  eval_result: ptr cstring = cast[ptr cstring](ptrInitializer)
  names: array[2, cstring] = [cstring("train"), cstring("test")]
  eval_names: ptr cstring = cast[ptr cstring](names.unsafeAddr())
  evalPattern: re.Regex = re.re(r".*:(0\.\d+).*:(0\.\d+).*")   # pick out auc values
var
  train_auc: seq[float64] = newSeqOfCap[float64](num_of_iterations)
  test_auc: seq[float64] = newSeqOfCap[float64](num_of_iterations)
  evalOut: string
  re_matches: array[2, string]
for i in 0..<num_of_iterations:
  # Update the model performance for each iteration
  safe_xgboost: XGBoosterUpdateOneIter(booster, i, train)

  # Give the statistics for the learner for training & testing dataset in terms
  # of error after each iteration
  safe_xgboost: XGBoosterEvalOneIter(booster, i, eval_dmats, cast[ptr cstring](eval_names),
    eval_dmats_size, eval_result)
  evalOut = $eval_result[]   # initializing here apparently only runs once
  stdout.writeline("$1" % [evalOut])

  # store the train and test auc in a seq (there's got to be a function that
  # makes this easier I would think)
  discard re.match(evalOut, evalPattern, re_matches, 2)
  train_auc.add(parseFloat(re_matches[0]))
  test_auc.add(parseFloat(re_matches[1]))

  # check easy early stopping criteria
  if i >= early_stopping_rounds and test_auc[^1] <= test_auc[^check_index]:
    break
echo()

# write out loss history data
echo "Writing training history loss to file..."
let f: File = open("training_history.csv", fmWrite)
var lineStr: string
for i in 0..high(train_auc):
  lineStr = "$1,$2" % [&"{train_auc[i]:.6f}", &"{test_auc[i]:.6f}"]
  f.writeLine(lineStr)
f.close()
echo()

# save the model (not in the C API example)
safe_xgboost: XGBoosterSaveModel(booster, "xgb_model_nim.xgb")

# get feature names - realize that we already have these as they were set above.
# But want to check that the 'get' function works properly
let
  feature_names_ptr: ptr ptr cstring = cast[ptr ptr cstring](ptrInitializer)
  g: File = open("feature_names.txt", fmWrite)
var feature: cstring
safe_xgboost: XGDMatrixGetStrFeatureInfo(train, "feature_name", outlen.addr(), feature_names_ptr)
echo "Number of features in dataset: $1" % [$outlen]
echo "Feature Names:"
for i in 0..<outlen:
  feature = cast[ptr cstring](cast[uint](feature_names_ptr[]) + uint(sizeof(ptr cstring)) * i)[]
  echo &"Feature {i:d}: {feature}"
  g.writeLine(feature)
g.close()
echo()


# 7. Predict the result on the test set using XGBoosterPredict
var
  output_length: bst_ulong
  output_result: ptr cfloat = cast[ptr cfloat](ptrInitializer)

safe_xgboost: XGBoosterPredict(booster, test, 0, 0, 0, output_length.addr(),
  output_result.addr())

echo "First 10 predictions on the test set..."
for i in 0..<output_length:
  var value: cfloat = cast[ptr cfloat](cast[uint](output_result) + uint(sizeof(cfloat)) * i)[]
  echo "prediction[$1] = $2" % [$i, &"{value:.5f}"]
  if i >= 10:
    break


# 8. Free all the internal structure used in your code using XGDMatrixFree and
# XGBoosterFree. This step is important to prevent memory leak.
echo "\nFreeing up memory resources..."
safe_xgboost: XGDMatrixFree(train)
safe_xgboost: XGDMatrixFree(test)
safe_xgboost: XGBoosterFree(booster)

# also free Nim memory where necessary (e.g., numpy buffers)
X_train_buffer.release()
X_test_buffer.release()
y_train_buffer.release()
y_test_buffer.release()

# and the allocated pointer intializer
dealloc(ptrInitializer)

# Skip 9
# 9. Get the number of features in your dataset using XGBoosterGetNumFeature.
