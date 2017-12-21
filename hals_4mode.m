function D = hals_4mode(Yt,A,B,C,R,maxiters)

 % FAST HALS NTF (Algorithm 7.5)
 Uinit_unorm = rand(size(Yt,4),R);
 D = bsxfun(@rdivide,Uinit_unorm,sqrt(sum(Uinit_unorm.^2,1)));
 T1 = D'*D;
% T2 = mttkrp(X,Z,n);
 
Y4 = reshape(permute(Yt,[4 1 2 3]),size(Yt,4),size(Yt,1)*size(Yt,2)*size(Yt,3));
T2 =(Y4*kr(C,kr(B,A)));
T3 = (A'*A).*(B'*B).*(C'*C);
res = []; 

 for iter = 1:maxiters
             
     % Loop for j
     for j = 1:R
         D(:,j) = D(:,j) + (T2(:,j) - D*T3(:,j))/T3(j,j);
     end
         
 end  
 
end