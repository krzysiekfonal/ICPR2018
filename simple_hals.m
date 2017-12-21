function [UC, UA, UB, output] = simple_hals(Y, X, J, JC, tol, MaxIter)

%Size
DimY = size(Y);
N = length(DimY);

%Initialization
UC = cell([N 1]);
UA = cell([N 1]);
UB = cell([N 1]);
GY = neye([J*ones(1,N)]);
GX = neye([J*ones(1,N)]);
normY = sqrt(sum(Y(:).^2));
normX = sqrt(sum(X(:).^2));

for n = 1:N
    UA{n} = rand(DimY(n),J);
    UA{n} = UA{n}.*repmat(1./sqrt(eps+sum(UA{n}.^2,1)),size(UA{n},1),1);
    UB{n} = rand(DimY(n),J);
    UB{n} = UB{n}.*repmat(1./sqrt(eps+sum(UB{n}.^2,1)),size(UB{n},1),1);
end

Y_hat = ntimes(GY,UA{1},1,2);
X_hat = ntimes(GX,UB{1},1,2);
for n = 2:N
    Y_hat = ntimes(Y_hat,UA{n},1,2); % tensor N-way
    X_hat = ntimes(X_hat,UB{n},1,2); % tensor N-way
end
EY = Y - Y_hat;
EX = X - X_hat;

% Main loop
for k = 1:MaxIter

    for j = 1:J
        
        Yj_hat = product_rank_1(UA,j,DimY);
        Yj = EY + Yj_hat;
        Xj_hat = product_rank_1(UB,j,DimY);
        Xj = EX + Xj_hat;
  
        for n = 1:N 
            
            Yn = reshape(permute(Yj,[n [1:n-1,n+1:N]]),DimY(n),prod(DimY)/DimY(n));
            Xn = reshape(permute(Xj,[n [1:n-1,n+1:N]]),DimY(n),prod(DimY)/DimY(n));
            WY = kr_minus_n_only_j(UA,n,j);
            WX = kr_minus_n_only_j(UB,n,j);
            UA{n}(:,j) = Yn*WY; %max(0,Yn*WY);
            UB{n}(:,j) = Xn*WX; %max(0,Yn*WY);
               
            if j <= JC
                t = UA{n}(:,j) + UB{n}(:,j);
                UA{n}(:,j) = t;
                UB{n}(:,j) = t;
            end
            
            if(n ~= N)
                UA{n}(:,j) = UA{n}(:,j)./sqrt(sum(UA{n}(:,j).^2 + eps));
                UB{n}(:,j) = UB{n}(:,j)./sqrt(sum(UB{n}(:,j).^2 + eps));
            end
        
        end % for n
        
        Yj_hat = product_rank_1(UA,j,DimY);
        EY = Yj - Yj_hat;
        Xj_hat = product_rank_1(UB,j,DimY);
        EX = Xj - Xj_hat;
        
    end % for j
       
        % Residual error
        Y_hat = ntimes(GY,UA{1},1,2);
        X_hat = ntimes(GX,UB{1},1,2);
        for n = 2:N
            Y_hat = ntimes(Y_hat,UA{n},1,2); % tensor N-way            
            X_hat = ntimes(X_hat,UB{n},1,2); % tensor N-way            
        end
        RY = Y - Y_hat; % residual matrix;
        RX = X - X_hat; % residual matrix;
        resY(k) = sqrt(sum(RY(:).^2))/normY;
        resX(k) = sqrt(sum(RX(:).^2))/normX;
        res(k) = (resY(k) + resX(k)) / 2;
      
        if ~mod(k,5)
            disp(['Iteration: ',num2str(k), ', Residual error Y: ',num2str(resY(k)),', Residual error X: ',num2str(resX(k))]);
        end
        
end % for k
for n = 1:N
    UC{n} = UA{n}(:,1:JC);
    UA{n} = UA{n}(:,JC+1:J);
    UB{n} = UB{n}(:,JC+1:J);
end
output.normresidual = res;

end
