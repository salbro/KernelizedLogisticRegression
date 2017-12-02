module LogReg

export batchgd, sgd, minibatchgd

using KernLogRegUtils;


function batchgd(X, y, λ, n_epochs, γ=0.01, ϵ=0.01, kern="gaussian")    
    n = size(X)[1]
    if kern == "gaussian"
        σ = get_σ(X)
        k = give_gaus_kern(σ)
        
    elseif kern == "linear"
        k = dot
    end
    
    K = create_kernel_matrix(k, X)
            
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
            tot += k(X[i,:], z)*c[i]
        end

        return 1.0 / (1 + e^(-tot))
    end
    
    return prob_predictor
end


function sgd(X, y, λ, n_epochs, γ=nothing, ϵ=0.01, kern="gaussian")    
    n = size(X)[1]
    if kern == "gaussian"
        σ = get_σ(X)
        k = give_gaus_kern(σ)
        
    elseif kern == "linear"
        k = dot
    end
    
    K = create_kernel_matrix(k, X)
    
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
            tot += k(X[i,:], z)*c[i]
        end

        return 1.0 / (1 + e^(-tot))
    end
    
    return prob_predictor
end


function minibatchgd(X, y, λ, n_epochs, batch_size, γ=0.01, ϵ=0.01, kern="gaussian")    
    n = size(X)[1]
    if kern == "gaussian"
        σ = get_σ(X)
        k = give_gaus_kern(σ)
        
    elseif kern == "linear"
        k = dot
    end
    
    K = create_kernel_matrix(k, X)
    
#     γ = get_γ(K, λ)
        
    c = zeros(n)
        
    for epoch in 1:n_epochs
        inds = sort(randperm(n)[1:batch_size])
        c[inds] = c[inds] + γ*(y[inds] - sigmoid(K[inds,:]*c) + 2*λ.*c[inds])
    end
    
    function prob_predictor(z)
        tot = 0
        for i in 1:n
            tot += k(X[i,:], z)*c[i]
        end

        return 1.0 / (1 + e^(-tot))
    end
    
    return prob_predictor
end


end


