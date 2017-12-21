function W = kr_minus_n_new(U,n)

N = size(U,1);

W = ones(1,size(U{1},2));
for i = [1:n-1,n+1:N]
    W = kr(U{i},W); 
end
    
