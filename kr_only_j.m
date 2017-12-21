function W = kr_only_j(U,j)

N = size(U,1);

W = 1;
for i = 1:N
    W = kr(U{i}(:,j),W); 
end
    
