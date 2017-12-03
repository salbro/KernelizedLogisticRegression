module LogReg

export batchgd, sgd, minibatchgd

using KernLogRegUtils;

"""
Description: performs batch gradient descent for the log-likelihood objective function
Inputs:
    X: the (n x p) training data matrix (rows are points)
    y: the (n x 1) vector of training labels {0,1}
    λ: L2 regularization parameter
    n_epochs: the number of gradient descent iterations
    γ: the learning rate
    ϵ: termination threshold; GD will stop early if objective value doesn't stop by > ϵ%
    kernel: the kernel function (default linear kernel)

Outputs:
    prob_predictor: a function which maps an input data point to a probability that its label is 1

"""
function batchgd(X, y, λ=0, n_epochs=1000, γ=0.01, ϵ=0.01, kernel=dot)    
    n = size(X)[1]

    if kernel==dot
        K = transpose(X)*X
    else
        K = create_kernel_matrix(kernel, X)
    end
            
    c = zeros(n)
    objectives = zeros(n_epochs)
    
    c = c + γ*(y - sigmoid(K*c) + 2*λ.*c)
    objectives[1] = cost(c, K, y)
    
    epochs_used = n_epochs
    
    for epoch in 2:n_epochs
        c = c + γ.*(y - sigmoid(K*c) + 2*λ.*c)

        objectives[epoch] = cost(c, K, y)
        if abs(objectives[epoch] - objectives[epoch-1])/objectives[epoch] < ϵ
            epochs_used = epoch
            break
        end
    end
    
    function prob_predictor(z)
        tot = 0
        for i in 1:n
            tot += kernel(X[i,:], z)*c[i]
        end

        return 1.0 / (1 + e^(-tot))
    end
    
    return prob_predictor
end

"""
Description: performs stochastic gradient descent for the log-likelihood objective function
Parameters and output same as above.
γ will be calculated if not provided
"""
function sgd(X, y, λ=0, n_epochs=1000, γ=nothing, ϵ=0.01, kernel=dot)    
    n = size(X)[1]

    
    if kernel==dot
        K = transpose(X)*X
    else
        K = create_kernel_matrix(kernel, X)
    end
    
    if γ == nothing
        γ = get_γ(K, λ)
    end
        
    c = zeros(n)

    Kc =  K*c
   
    for epoch in 1:n_epochs
        i = rand(1:n)
        Kc_i_old = K[:,i].*c[i]
        c[i] = c[i] + γ.*(y[i] - sigmoid(Kc[i]) + 2*λ.*c[i])
        
        # update Kc O(n)
        Kc = Kc - Kc_i_old + K[:,i].*c[i]
    end
    
    
    function prob_predictor(z)
        tot = 0
        for i in 1:n
            tot += kernel(X[i,:], z)*c[i]
        end

        return 1.0 / (1 + e^(-tot))
    end
    
    return prob_predictor
end


"""
Description: performs minibatch gradient descent for the log-likelihood objective function
Parameters and output same as above, with one addition:
    batch_size: the size of each batch during gradient descent (recommend batch_size around 30)
"""
function minibatchgd(X, y, λ=0, n_epochs=1000, batch_size, γ=0.01, ϵ=0.01, kernel=dot)    
    n = size(X)[1]
  
    if kernel==dot
        K = transpose(X)*X
    else
        K = create_kernel_matrix(kernel, X)
    end
    
    c = zeros(n)
        
    for epoch in 1:n_epochs
        inds = sort(randperm(n)[1:batch_size])
        c[inds] = c[inds] + γ*(y[inds] - sigmoid(K[inds,:]*c) + 2*λ.*c[inds])
    end
    
    function prob_predictor(z)
        tot = 0
        for i in 1:n
            tot += kernel(X[i,:], z)*c[i]
        end

        return 1.0 / (1 + e^(-tot))
    end
    
    return prob_predictor
end


end


