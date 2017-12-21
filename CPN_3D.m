function [C,delta,elapsed_time] = CP_3D(Yr,Yt,Class_train_inx,Class_test_inx,J,MCRuns,MaxIter)

% TRAINING
% ================================================================
alpha = 1e-6;
C = zeros(max(Class_test_inx)); % initialization
opts = cpo_als2;
opts.maxiters = 5*MaxIter;
opts.orthomodes = [3];
opts.init = 'nvecs';
for k = 1:MCRuns
  
     tic
%     [P,output] = cpo_als2(tensor(Yr),J,opts);
%     Ar = P.U{1}; Br = P.U{2}; Cr = P.U{3};
%   %  [Ar,Br,Cr] = cp_als_3D(Yr,J,MaxIter);
%     elapsed_time = toc;
% 
%     % TESTING
%     % ================================================================
%     Y3 = reshape(permute(Yt,[3 1 2]),size(Yt,3),size(Yt,1)*size(Yt,2));
% 
%     At = Ar'*Ar;
%     Bt = Br'*Br;
%     
%     Ct =(Y3*kr(Br,Ar))*inv(Bt.*At + alpha*eye(J));
%     Ct = Ct.*repmat(1./sqrt(sum(Ct.^2,2)+eps),1,size(Ct,2));
    
    Ycc = cat(3,Yr,Yt);
    [P,output] = cp_fhals(tensor(Ycc),J,opts);
    Cr = P.U{3}(1:size(Yr,3),:);
    Ct = P.U{3}(size(Yr,3)+1:end,:);
     elapsed_time = toc;

    
    Class_knn = knnclassify(Ct,Cr,Class_train_inx,1,'cosine');
    delta = 100*(length(find((Class_knn - Class_test_inx)==0))/length(Class_test_inx));

    % Macierz prawd
    T = zeros(max(Class_test_inx),length(Class_test_inx));
    Ts = T;
    I = eye(max(Class_test_inx));
    for i = 1:length(Class_test_inx)
        T(:,i) = I(:,Class_test_inx(i));
        Ts(:,i) = I(:,Class_knn(i));
    end
    [Cx,rate]=confmat(Ts',T');
    C = C + Cx;

end
C = C/MCRuns;

end
