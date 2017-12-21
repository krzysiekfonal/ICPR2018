function C = hals_3mode(Yt,A,B,R,maxiters)

 % FAST HALS NTF (Algorithm 7.5)
 Uinit_unorm = rand(size(Yt,3),R);
 C = bsxfun(@rdivide,Uinit_unorm,sqrt(sum(Uinit_unorm.^2,1)));
 T1 = C'*C;
% T2 = mttkrp(X,Z,n);
 
Y3 = reshape(permute(Yt,[3 1 2]),size(Yt,3),size(Yt,1)*size(Yt,2));
T2 =(Y3*kr(B,A));
T3 = (A'*A).*(B'*B);
res = []; 

 for iter = 1:maxiters
             
     % Compute T3
   %  T3 = T1./(Z{n}'*Z{n});
         
     % Loop for j
     for j = 1:R
         C(:,j) = C(:,j) + (T2(:,j) - C*T3(:,j))/T3(j,j);
      %   C(:,j) = C(:,j)/norm(C(:,j),2);
     end

    % C = C*diag(1./sqrt(sum(C.^2,1)));
    % C = diag(1./sqrt(sum(C.^2,2)))*C;
     
%      Ir = neye([R R R]);
%      Y_hat = ntimes(ntimes(ntimes(Ir,A,1,2),B,1,2),C,1,2); % tensor 3-way
%      res(iter) = sum(sum(sum((Yt - Y_hat).^2)));
         
 end  
 
% figure
% plot(res)

end