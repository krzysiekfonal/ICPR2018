function [C,delta,elapsed_time,res] = CP_4D(Yr,Yt,Class_train_inx,Class_test_inx,J,MCRuns,Tol,MaxIter)

% TRAINING
% ================================================================
alpha = 1e-6;
C = zeros(max(Class_test_inx)); % initialization
for k = 1:MCRuns
  
    tic
    [Ar,Br,Cr,Dr,res] = cp_als_4D(Yr,J,'tol',Tol,'maxiters',MaxIter,'printitn',1);
    elapsed_time = toc;

    % TESTING
    % ================================================================
    Y3 = reshape(permute(Yt,[4 1 2 3]),size(Yt,4),size(Yt,1)*size(Yt,2)*size(Yt,3));

    Atr = Ar'*Ar;
    Btr = Br'*Br;
    Ctr = Cr'*Cr;
     
    Dt =(Y3*kr(Cr,kr(Br,Ar)))*inv(Ctr.*Btr.*Atr + alpha*eye(J));
    Dt = Dt.*repmat(1./sqrt(sum(Dt.^2,2)+eps),1,size(Dt,2));
    Class_knn = knnclassify(Dt,Dr,Class_train_inx,1,'correlation');
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
