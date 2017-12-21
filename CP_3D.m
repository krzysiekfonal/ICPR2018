function [C,delta,elapsed_time,res] = CP_3D(Yr,Yt,Class_train_inx,Class_test_inx,J,MCRuns,Tol,MaxIter)

% TRAINING
% % ================================================================
% opts = cp_init;
% opts.maxiters = MaxIter;
% opts.init = {'nvecs' 'rand'};
% opts.printitn = 1;

alpha = 1e-6;
C = zeros(max(Class_test_inx)); % initialization
for k = 1:MCRuns
  
    tic
    [Ar,Br,Cr,res] = cp_als_3D(Yr,J,'tol',Tol,'maxiters',MaxIter,'printitn',1);
    elapsed_time = toc;

    % TESTING
    % ================================================================
    Y3 = reshape(permute(Yt,[3 1 2]),size(Yt,3),size(Yt,1)*size(Yt,2));

    At = Ar'*Ar;
    Bt = Br'*Br;
    
    Ct =(Y3*kr(Br,Ar))*inv(Bt.*At + alpha*eye(J));
    Ct = Ct.*repmat(1./sqrt(sum(Ct.^2,2)+eps),1,size(Ct,2));
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
