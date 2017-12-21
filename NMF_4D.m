function [C,delta,elapsed_time,res] = NMF_4D(Yr,Yt,Class_train_inx,Class_test_inx,J,MCRuns,Tol,MaxIter)

DimY = size(Yr);
Yr_mtx = reshape(Yr,[DimY(1)*DimY(2)*DimY(3),DimY(4)]);
Yt_mtx = reshape(Yt,[DimY(1)*DimY(2)*DimY(3),size(Yt,4)]); 

if min(min(Yr_mtx)) < 0
   Yr_mtx = Yr_mtx - min(min(Yr_mtx)) + eps; 
end

if min(min(Yt_mtx)) < 0
   Yt_mtx = Yt_mtx - min(min(Yt_mtx)) + eps; 
end

% TRAINING
% ================================================================
alpha = 1e-6;
res = [];

C = zeros(max(Class_test_inx)); % initialization
for k = 1:MCRuns
  
    A = rand(size(Yr_mtx,1),J);
    X = rand(J,size(Yr_mtx,2));
    
    tic
    [Ar,Xr,res] = nmf_fast_hals(Yr_mtx,A,X,1e-8,Tol,MaxIter,1);
    elapsed_time = toc;

    % TESTING
    % ================================================================
  
    Xt = rand(J,size(Yt_mtx,2));
    Xt = fast_hals(Ar,Yt_mtx,Xt,300);
    Class_knn = knnclassify(Xt',Xr',Class_train_inx,1,'cosine');
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
