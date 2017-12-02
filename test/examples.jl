include("../src/KernLogRegUtils.jl")
include("../src/LogReg.jl");

using LogReg, KernLogRegUtils, Plots, DataFrames, RDatasets

function get_XY(csv)
    df = readtable(csv)
    return Matrix(df[2:end-1]), Array(df[:label])
end;

# non-linearly separable data 
X,y = get_XY("data/gaus_2D_train.csv");
Xte, yte = get_XY("data/gaus_2D_test.csv");


####################### BATCH Gradient Descent ####################
predictor_function = batchgd(X, y, 0, 200, 1);
batch_score, preds = evaluate(Xte, yte, predictor_function)
print("batch_score, Gaussian Kernel: ", batch_score)


predictor_function = batchgd(X, y, 0, 200, 1, 0.01, "linear");
batch_score, preds = evaluate(Xte, yte, predictor_function)
print("batch_score, Linear Kernel: ", batch_score)
###################################################################


################### Stochastic Gradient Descent ###################
predictor_function = sgd(X, y, 0, 2000);
sgd_score, preds = evaluate(Xte, yte, predictor_function)
print("stochastic test accuracy: ", sgd_score)
###################################################################


################# Minibatch Gradient Descent ######################
predictor_function = minibatchsgd(X, y, 0, 500, 30);
minbatch_score, preds = evaluate(Xte, yte, predictor_function)
print("minibatch test accuracy: ", minbatch_score)
###################################################################




################ Generating Test-Set Predictions ##################
predictions = predict(Xte, predictor_function)




