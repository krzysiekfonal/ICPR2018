function Y = ttv_arrays(X,U)

% X: N-way input tensor,
% U: cell array of N vectors 

DimX = size(X);
N = length(DimX);
Xz = X;

for n = 1:N
    
    DimXz = size(Xz);
    Xz1 = reshape(Xz,DimXz(1),prod(DimXz)/DimXz(1)); % unfolding along 1-th mode
    Y = U{n}'*Xz1;
    if n < N
       Xz = reshape(Y,[DimX(n+1) length(Y)/DimX(n+1)]); % matricization along the n-th mode
    end    
end