function UC = hals_4mode_ff(Y,X,UC,UA,UB,RC,alpha,maxiters,printitn)

%% Extract number of dimensions and norm of X and Y.
dim = size(Y);
N = ndims(Y);
n = 4;

%% init U matrices and other params
beta = 1 - alpha;
UCinit_unorm = rand(size(Y,n),RC);
UC{n} = bsxfun(@rdivide,UCinit_unorm,sqrt(sum(UCinit_unorm.^2,1)));

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

% Calculate Yd and Xd as a difference between Y,X and individual
% parts
Yd = Yn - UA{n} * krUAexcn';
Xd = Xn - UB{n} * krUBexcn';

% Calculate Unew = X_(n) * khatrirao(all U except n, 'r').
T2 = (alpha .* Yd + beta .* Xd) * krUCexcn;

% Calculate T3
T3 = ones(RC,RC);
for i = 1:N
    if i~=n
        T3 = T3 .* (UC{i}'*UC{i});        
    end
end

for iter = 1:maxiters

    % Loop for j for common part
    for j = 1:RC
        UC{n}(:,j) = UC{n}(:,j) + T2(:,j) - UC{n}*T3(:,j);                
    end    
end   

end


