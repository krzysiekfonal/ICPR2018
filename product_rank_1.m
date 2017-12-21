function Yj_hat = product_rank_1(U,j,DimY)

N = size(U,1);

Uj = U{N}(:,j);
for n = N:-1:2
    Uj = kron(Uj,U{n-1}(:,j)); % tensor N-way of rank 1            
end
Yj_hat = reshape(Uj,DimY);