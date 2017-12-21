function [A,B,C,D,G,Y_hat] = tucker_hosvd(Y,J)

% Size
DimY = size(Y);

% Unfolding
Y1 = reshape(Y,DimY(1),DimY(2)*DimY(3)*DimY(4));
Y2 = reshape(permute(Y,[2 1 3 4]),DimY(2),DimY(1)*DimY(3)*DimY(4));
Y3 = reshape(permute(Y,[3 1 2 4]),DimY(3),DimY(1)*DimY(2)*DimY(4));
Y4 = reshape(permute(Y,[4 1 2 3]),DimY(4),DimY(1)*DimY(2)*DimY(3));
       

% Tensor decomposition
[E1,D1] = eig(Y1*Y1');
A = fliplr(E1(:,DimY(1)-J(1)+1:DimY(1)));

[E2,D2] = eig(Y2*Y2');
B = fliplr(E2(:,DimY(2)-J(2)+1:DimY(2)));
        
[E3,D3] = eig(Y3*Y3');
C = fliplr(E3(:,DimY(3)-J(3)+1:DimY(3)));
      
[E4,D4] = eig(Y4*Y4');
D = fliplr(E4(:,DimY(4)-J(4)+1:DimY(4)));
              
G = ntimes(ntimes(ntimes(ntimes(Y,A',1,2),B',1,2),C',1,2),D',1,2); % core tensor
Y_hat = ntimes(ntimes(ntimes(ntimes(G,A,1,2),B,1,2),C,1,2),D,1,2); % tensor 4-way

D = D.*repmat(1./sqrt(sum(D.^2,2)+eps),1,size(D,2));

