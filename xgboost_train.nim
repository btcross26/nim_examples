include xgboost_wrapper

import strutils

# use a Nim template to (for the most part) mimic the #define safe_xgboost stmt
# in the XGBoost C API code
template safe_xgboost(procCall: untyped) =
  let err: int = cast[int](procCall)   # cast cint to int
  if err != 0:
    let pos = instantiationInfo()             # to get the current file and line number
    # use stderr in Nim to write to stderr, as opposed to fprintf in safe_xgboost
    stderr.writeLine("$1:$2: error in $3: $4\n" % [pos.filename, $pos.line,
      procCall.repr, $XGBGetLastError()])
    quit(1)   # replaces C++ exit(1)

# compiler options to include
# -d:macOS
# --cincludes:$HOME/anaconda3/envs/nim_demo/include
# --clibdir:$HOME/anaconda3/envs/nim_demo/lib
# --clib:xgboost
# --passL:"-rpath $HOME/anaconda3/envs/nim_demo/lib"
# the code below is taken directly from the xgboost C API example and modified
# accordingly to work with Nim syntax and the example dataset
# https://xgboost.readthedocs.io/en/latest/tutorials/c_api_tutorial.html

# 1. If the dataset is available in a file, it can be loaded into a DMatrix
#    object using the XGDMatrixCreateFromFile
var
  train: DMatrixHandle
  test: DMatrixHandle

# Load the data from files & store it in data variable of DMatrixHandle datatype
# Assumes that the datafiles are in the current working directory
safe_xgboost: XGDMatrixCreateFromFile("/path/to/file/filename", 1, train)
safe_xgboost: XGDMatrixCreateFromFile("path to test data", 1, test)

# 2. Skip this step as loading data from file
# 3. Create a Booster object for training & testing on dataset using XGBoosterCreate
var booster: BoosterHandle
const eval_dmats_size: bst_ulong = 2   # const means this gets evaluated at compiletime
# We assume that training and test data have been loaded into 'train' and 'test'
let train_test_array: array[eval_dmats_size, DMatrixHandle] = [train, test]
# this was tricky, but arrays are never pointers in Nim, so to get the interface
# right and get used to this will take some work
let eval_dmats = cast[ptr UncheckedArray[DMatrixHandle]](unsafeAddr(train_test_array[0]))
# DMatrixHandle eval_dmats[eval_dmats_size] = {train, test};
# a little bit of nimfu going on below - getting unsafe pointer to train_test_array to create UncheckedType
safe_xgboost: XGBoosterCreate(eval_dmats, eval_dmats_size, booster.unsafeAddr())

# # 4. For each DMatrix object, set the labels using XGDMatrixSetFloatInfo. Later
# # you can access the label using XGDMatrixGetFloatInfo.
# const int ROWS=5, COLS=3;
# const int data[ROWS][COLS] = { {1, 2, 3}, {2, 4, 6}, {3, -1, 9}, {4, 8, -1}, {2, 5, 1}, {0, 1, 5} };
# DMatrixHandle dmatrix;
#
# safe_xgboost: XGDMatrixCreateFromMat(data, ROWS, COLS, -1, &dmatrix)
#
# // variable to store labels for the dataset created from above matrix
# float labels[ROWS];
#
# for (int i = 0; i < ROWS; i++) {
#   labels[i] = i;
# }
#
# // Loading the labels
# safe_xgboost: XGDMatrixSetFloatInfo(dmatrix, "labels", labels, ROWS)
#
# // reading the labels and store the length of the result
# bst_ulong result_len;
#
# // labels result
# const float *result;
#
# safe_xgboost: XGDMatrixGetFloatInfo(dmatrix, "labels", &result_len, &result)
#
# for(unsigned int i = 0; i < result_len; i++) {
#   printf("label[%i] = %f\n", i, result[i]);
# }
#
#
# # 5. Set the parameters for the Booster object according to the requirement
# # using XGBoosterSetParam . Check out the full list of parameters available
# # here.
# BoosterHandle booster;
# safe_xgboost: XGBoosterSetParam(booster, "booster", "gblinear")
# safe_xgboost: XGBoosterSetParam(booster, "max_depth", "3")
# # default eta  = 0.3
# safe_xgboost: XGBoosterSetParam(booster, "eta", "0.1")
#
# # 6. Train & evaluate the model using XGBoosterUpdateOneIter and
# # XGBoosterEvalOneIter respectively.
# let num_of_iterations: int = 20
# const char* eval_names[eval_dmats_size] = {"train", "test"};
# const char* eval_result = NULL;
#
# for (int i = 0; i < num_of_iterations; ++i) {
#   // Update the model performance for each iteration
#   safe_xgboost: XGBoosterUpdateOneIter(booster, i, train)
#
#   // Give the statistics for the learner for training & testing dataset in terms of error after each iteration
#   safe_xgboost: XGBoosterEvalOneIter(booster, i, eval_dmats, eval_names, eval_dmats_size, &eval_result)
#   printf("%s\n", eval_result);
# }
#
# # 7. Predict the result on a test set using XGBoosterPredict
# var
#   output_length: bst_ulong
#   output_result: ptr float
#
# safe_xgboost: XGBoosterPredict(booster, test, 0, 0, &output_length, &output_result)
#
# for i in 0..<output_length:
#   echo "prediction[%i] = %f \n", i, output_result[i])
#
# # 8. Free all the internal structure used in your code using XGDMatrixFree and
# # XGBoosterFree. This step is important to prevent memory leak.
# safe_xgboost: XGDMatrixFree(dmatrix)
# safe_xgboost: XGBoosterFree(booster)
#
# # 9. Get the number of features in your dataset using XGBoosterGetNumFeature.
#
# var num_of_features: bst_ulong = 0
#
# // Assuming booster variable of type BoosterHandle is already declared
# // and dataset is loaded and trained on booster
# // storing the results in num_of_features variable
# safe_xgboost: XGBoosterGetNumFeature(booster, num_of_features)
#
# // Printing number of features by type conversion of num_of_features variable from bst_ulong to unsigned long
# printf("num_feature: %lu\n", (unsigned long)(num_of_features));
#
# # Save the model (not in example)
