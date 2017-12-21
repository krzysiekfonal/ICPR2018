function [A,X,res] = nmf_hals_fast(Y,A,X,tol_proj,tol_obj,MaxIter,show_inx)

gradA = A*(X*X') - Y*X'; gradX = (A'*A)*X - A'*Y;
initgrad = norm([gradA; gradX'],'fro');
if show_inx
    fprintf('Init gradient norm %f\n', initgrad); 
end
tolA = max(0.00001,tol_proj)*initgrad; tolX = tolA;
res(1) = norm(Y - A*X,'fro')/norm(Y,'fro');
R = size(A,2); k_inner = 10;
fit = 0;

    for k = 1:MaxIter

        fitold = fit;    
        projnorm = norm([gradA(gradA<0 | A>0); gradX(gradX<0 | X>0)]);
        if show_inx & (k < 10 | ~mod(k,10))
           disp(['FAST-HALS iterations: ',num2str(k), ',  Gradient norm: ',num2str(projnorm )]);
        end

      % Stopping criterion  
        if projnorm < tol_proj*initgrad,
           break;
        end
                
      % Update for X
        [X,gradX,iterX] = fast_hals_inner(A,Y,X,R,tolX,k_inner); 
        if iterX == 1
           tolX = 0.1*tolX;
        end
        
% %       % Normalization
        dX = eps+(sum(X,2));
        X = bsxfun(@ldivide,dX,X);
        A = bsxfun(@times,dX',A);

      % Update for A  
        [At,gradA,iterA] = fast_hals_inner(X',Y',A',R,tolA,k_inner);
        A = At'; gradA = gradA';
        if iterA == 1
           tolA = 0.1*tolA;
        end

% %       % Normalization
        dA = eps+(sum(A,1));
        A = bsxfun(@rdivide,A,dA);
        X = bsxfun(@times,dA',X);

        res(k+1) = norm(Y - A*X,'fro')/norm(Y,'fro');
        fit = 1 - res(k+1);
        fitchange = abs(fitold - fit);
         
       % Stagnation 
         if (k > 10) && (fitchange < tol_obj)
             disp(['Stagnation of FAST-HALS iterations: ',num2str(k+1), ', Fit: ',num2str(fit), ', Fit_change: ', num2str(fitchange)]);
             break;
         end

    end % for k
    
end

% ========================================================================
function [X,G,iter] = fast_hals_inner(A,Y,X,r,tol,k_inner)

    W = A'*Y; V = A'*A; 
    for iter = 1:k_inner

      % stopping
       G = V*X - W;
       projgrad = norm(G(G < 0 | X > 0));
       if projgrad < tol,
          break
       end

        for j = 1:r 
            X(j,:) = max(eps,X(j,:) + (W(j,:) - V(j,:)*X)/V(j,j));  
        end
              
    end % for k
end % function FAST-HALS_inner




