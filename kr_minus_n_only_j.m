function W = kr_minus_n_only_j(U,n,j)

N = size(U,1);

W = 1;
for i = [1:n-1,n+1:N]
    W = kr(U{i}(:,j),W); 
end
    
