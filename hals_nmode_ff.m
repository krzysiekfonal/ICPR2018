function [UC, UA, UB] = hals_nmode_ff(Y,X,UC,UA,UB,RC,RI,n,alpha,maxiters,printitn)

%% Extract number of dimensions and norm of X and Y.
dim = size(Y);
N = ndims(Y);
%normY = norm(Y);
%normX = norm(X);

%% init U matrices and other params
beta = 1 - alpha;
UCinit_unorm = rand(size(Y,n),RC);
UC{n} = zeros(size(UCinit_unorm));
for j=1:RC
    UC{n}(:,j) = UCinit_unorm(:,j)/(norm(UCinit_unorm(:,j)) + eps);
end
%UC{n} = bsxfun(@rdivide,UCinit_unorm,sqrt(sum(UCinit_unorm.^2,1)));
UAinit_unorm = rand(size(Y,n),RI);
UA{n} = zeros(size(UAinit_unorm));
for j=1:RI
    UA{n}(:,j) = UAinit_unorm(:,j)/(norm(UAinit_unorm(:,j)) + eps);
end
%UA{n} = bsxfun(@rdivide,UAinit_unorm,sqrt(sum(UAinit_unorm.^2,1)));
UBinit_unorm = rand(size(X,n),RI);
UB{n} = zeros(size(UBinit_unorm));
for j=1:RI
    UB{n}(:,j) = UBinit_unorm(:,j)/(norm(UBinit_unorm(:,j)) + eps);
end
%UB{n} = bsxfun(@rdivide,UBinit_unorm,sqrt(sum(UBinit_unorm.^2,1)));

if printitn>0
  fprintf('\nCP_HALS projection:\n');
end

%% Main Loop: Iterate until convergence

% FAST HALS NTF (Algorithm 7.5)

% Calculate unfolded X and Y
Yn = reshape(permute(double(Y),[n [1:n-1,n+1:N]]),dim(n),prod(dim)/dim(n)); 
Xn = reshape(permute(double(X),[n [1:n-1,n+1:N]]),dim(n),prod(dim)/dim(n)); 

% Calculate khatrirao(all U except n) for U->C,A,B
krUCexcn = kr_minus_n_new(UC,n);
krUAexcn = kr_minus_n_new(UA,n);
krUBexcn = kr_minus_n_new(UB,n);

% Calculate T3
T3C = ones(RC,RC);
T3A = ones(RI,RI);
T3B = ones(RI,RI);
for i = 1:N
    if i~=n
        T3C = T3C .* (UC{i}'*UC{i});
        T3A = T3A .* (UA{i}'*UA{i});
        T3B = T3B .* (UB{i}'*UB{i});
    end
end

for iter = 1:maxiters

    % Calculate Yd and Xd as a difference between Y,X and individual
    % parts
    Yd = Yn - UA{n} * krUAexcn';
    Xd = Xn - UB{n} * krUBexcn';

    % Calculate Unew = X_(n) * khatrirao(all U except n, 'r').
    T2C = (alpha .* Yd + beta .* Xd) * krUCexcn;
    
    % Loop for j for common part
    for j = 1:RC
        UC{n}(:,j) = UC{n}(:,j) + T2C(:,j) - UC{n}*T3C(:,j);                
    end
    
    % Calculate Yg and Xg as a difference between Y,X and common part
    Yg = Yn - UC{n} * krUCexcn';
    Xg = Xn - UC{n} * krUCexcn';
    
    % Calculate Unew = X_(n) * khatrirao(all U except n, 'r').    
    T2A = Yg * krUAexcn;
    T2B = Xg * krUBexcn;
    
    % Loop for j for individual part
    for j = 1:RI       
        UA{n}(:,j) = UA{n}(:,j) + T2A(:,j) - UA{n}*T3A(:,j);
        UB{n}(:,j) = UB{n}(:,j) + T2B(:,j) - UB{n}*T3B(:,j);                
    end
end   

end


