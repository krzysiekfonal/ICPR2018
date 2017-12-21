function X = fast_hals(A,Y,X,k_inner)

    r = size(A,2);
    W = A'*Y; V = A'*A; 
    for iter = 1:k_inner

        for j = 1:r 
            X(j,:) = max(eps,X(j,:) + (W(j,:) - V(j,:)*X)/V(j,j));  
        end
              
    end % for k
end % function FAST-HALS_inner
