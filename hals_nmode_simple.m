function [UC, UA, UB] = hals_nmode_simple(Y, X, UC,UA,UB, J, JC, n, MaxIter)

%Size
DimY = size(Y);
N = length(DimY);

%Initialization
GY = neye([J*ones(1,N)]);
GX = neye([J*ones(1,N)]);
normY = sqrt(sum(Y(:).^2));
normX = sqrt(sum(X(:).^2));

for i=1:N
    UA{i}=[UC{i} UA{i}];
    UB{i}=[UC{i} UB{i}];
end

UA{n} = rand(DimY(n),J);
UA{n} = UA{n}.*repmat(1./sqrt(eps+sum(UA{n}.^2,1)),size(UA{n},1),1);
UB{n} = rand(DimY(n),J);
UB{n} = UB{n}.*repmat(1./sqrt(eps+sum(UB{n}.^2,1)),size(UB{n},1),1);


Y_hat = ntimes(GY,UA{1},1,2);
X_hat = ntimes(GX,UB{1},1,2);
for i = 2:N
    Y_hat = ntimes(Y_hat,UA{i},1,2); % tensor N-way
    X_hat = ntimes(X_hat,UB{i},1,2); % tensor N-way
end
EY = Y - Y_hat;
EX = X - X_hat;

UAv=cell([N 1]);                 
UBv=cell([N 1]);

% Main loop
for k = 1:MaxIter
    

    for j = 1:J
        
        Yj_hat = product_rank_1(UA,j,DimY);
        Yj = EY + GY(j,j,j,j)*Yj_hat;
        Xj_hat = product_rank_1(UB,j,DimY);
        Xj = EX + GX(j,j,j,j)*Xj_hat;
        
        
            
            Yn = reshape(permute(Yj,[n [1:n-1,n+1:N]]),DimY(n),prod(DimY)/DimY(n));
            Xn = reshape(permute(Xj,[n [1:n-1,n+1:N]]),DimY(n),prod(DimY)/DimY(n));
            WY = kr_minus_n_only_j(UA,n,j);
            WX = kr_minus_n_only_j(UB,n,j);
            UA{n}(:,j) = GY(j,j,j,j)*Yn*WY; %max(0,Yn*WY);
            UB{n}(:,j) = GX(j,j,j,j)*Xn*WX; %max(0,Yn*WY);
               
            if j <= JC
                t = UA{n}(:,j) + UB{n}(:,j);
                UA{n}(:,j) = t;
                UB{n}(:,j) = t;
            end
            
            %if(n ~= N)
                UA{n}(:,j) = UA{n}(:,j)./sqrt(sum(UA{n}(:,j).^2 + eps));
                UB{n}(:,j) = UB{n}(:,j)./sqrt(sum(UB{n}(:,j).^2 + eps));
            %end
            
        for i=1:N    
            UAv{i} = UA{i}(:,j);
            UBv{i} = UB{i}(:,j);
        end
        
        GY(j,j,j,j)=ttv_arrays(Yj, UAv);
        GX(j,j,j,j)=ttv_arrays(Xj, UBv);
        
        Yj_hat = product_rank_1(UA,j,DimY);
        EY = Yj - GY(j,j,j,j)*Yj_hat;
        Xj_hat = product_rank_1(UB,j,DimY);
        EX = Xj - GX(j,j,j,j)*Xj_hat;
        
    end % for j
        
end % for k

    UC{n} = UA{n}(:,1:JC);
    UA{n} = UA{n}(:,JC+1:J);
    UB{n} = UB{n}(:,JC+1:J);

end
