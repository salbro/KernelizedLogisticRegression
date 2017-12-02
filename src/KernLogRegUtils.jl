module KernLogRegUtils

export give_gaus_kern, sigmoid, get_σ, get_γ, create_kernel_matrix, cost, predict, evaluate


function give_gaus_kern(s)
    function gaus_kern(x1,x2)
        return e^((-norm(x1-x2)^2)/(2*s^2))
    end
    return gaus_kern
end

function sigmoid(x)
    1 ./ (1 .+ e.^(-x))
end

# finds a good value of σ for the gaussian kernel. average distances of data points
function get_σ(X)
    n,p = size(X)
    tot = 0.0
    for i in 1:(size(X)[1]-1)
        tot += sum((sum((X[i+1:end,:] .- reshape(X[i,:], (1,p))).^2, 2)).^0.5)
    end
    avg = tot / ((n-1)*n/2.0)
    return avg
end

function get_γ(K, λ)
    n = size(K)[1]
    return 1.0 / maximum(eigvals(K./n + λ.*eye(n)))
end

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

function cost(c, K, y)
    return -sum(log.(sigmoid(y.*(K*c))))
end


function predict(test_x, f)
    n = size(test_x)[1]
    preds = zeros(n)
    for i in 1:n
        preds[i] = 1*(f(test_x[i,:])>0.5)
    end
    return preds
end

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