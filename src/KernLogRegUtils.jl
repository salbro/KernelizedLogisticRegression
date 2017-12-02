module KernLogRegUtils

export give_gaus_kern, sigmoid, get_σ, get_γ, create_kernel_matrix, cost, predict, evaluate

"""
Description: returns a gaussian kernel function for a given σ
Input: σ (a number)
Output: a gaussian (RBF) kernel function
"""
function give_gaus_kern(σ)
    function gaus_kern(x1,x2)
        return e^((-norm(x1-x2)^2)/(2*σ^2))
    end
    return gaus_kern
end

"""
Description: Vector-scalable, element-wise sigmoid squashing function
Inputs: x (vector or scalar)
"""
function sigmoid(x)
    1 ./ (1 .+ e.^(-x))
end

"""
Description: Finds a reasonable value for Gaussian kernel's σ parameter,
             based on average distance between points in data matrix
Inputs: X (data matrix)
"""
function get_σ(X)
    n,p = size(X)
    tot = 0.0
    for i in 1:(size(X)[1]-1)
        tot += sum((sum((X[i+1:end,:] .- reshape(X[i,:], (1,p))).^2, 2)).^0.5)
    end
    avg = tot / ((n-1)*n/2.0)
    return avg
end


"""
Description: Finds a reasonable step size for GD based on the maximum eigenvalue of the Hessian 
Inputs:
    K: kernel matrix
    λ: regularization parameter
Outputs:
    γ: step-size
"""
function get_γ(K, λ)
    n = size(K)[1]
    return 1.0 / maximum(eigvals(K./n + λ.*eye(n)))
end

"""
Description: Builds a kernel matrix from a kernel function and a data matrix
Inputs:
    kern: kernel function
    X: data matrix
Returns:
    K: kernel matrix
"""
function create_kernel_matrix(kern, X)
    n = size(X)[1]
    K = zeros((n,n))
    for i in 1:n
        for j in 1:n
            K[i,j] = kern(X[i,:], X[j,:])
        end
    end
    return K
end


"""
Description: Calculates the value of the log-likelihood objective function (as a cost)
             for an iteration of gradient descent
Inputs:
    c: the current value of the alternate weight vector c
    K: the kernel matrix
    y: the training labels (0, 1)
"""
function cost(c, K, y)
    return -sum(log.(sigmoid(y.*(K*c))))
end

"""
Description: Returns predictions for a test set and a predictor function
Inputs: 
    test_x: a list of test points, same dimension as training points
    f: a predict function which maps from input space into {0,1}
Outputs:
    preds: list of {0,1} predictions, one for each element in the test set
"""
function predict(test_x, f)
    n = size(test_x)[1]
    preds = zeros(n)
    for i in 1:n
        preds[i] = 1*(f(test_x[i,:])>0.5)
    end
    return preds
end


"""
Description: Calculates both the accuracy and the very predictions of a function on the test set
Inputs:
    test_x: a list of test points, same dimension as training points
    test_y: testing labels {0,1}
    f: a predict function which maps from input space into {0,1}
Returns: a tuple (accuracy, predictions)
    accuracy: the accuracy, between 0 and 1, of the fitted predictor on the test set
    preds: list of {0,1} predictions, one for each element in the test set
"""
function evaluate(test_x, test_y, f)
    tot = 0
    preds = zeros(length(test_y))
    n = size(test_x)[1]
    for i in 1:n
        preds[i] = 1*(f(test_x[i,:])>0.5)
        tot += 1*(preds[i] == test_y[i])
    end
    return tot*1.0 / n, preds
end



end