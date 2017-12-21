function [C,delta,elapsed_time] = Multiclass_NMF_4D(Yr,Yt,Class_train_inx,Class_test_inx,J,MCRuns,MaxIter)

Yr = permute(Yr,[4 3 2 1]);
Yt = permute(Yt,[4 3 2 1]);

DimYr = size(Yr);
DimYt = size(Yt);

Yr_tmp = permute(reshape(Yr,[DimYr(1)*DimYr(2),DimYr(3),DimYr(4)]),[3 2 1]);
Yr_mtx = reshape(Yr_tmp,size(Yr_tmp,1)*size(Yr_tmp,2),size(Yr_tmp,3));

Yt_tmp = permute(reshape(Yt,[DimYt(1)*DimYt(2),DimYt(3),DimYt(4)]),[3 2 1]);
Yt_mtx = reshape(Yt_tmp,size(Yt_tmp,1)*size(Yt_tmp,2),size(Yt_tmp,3));

if min(min(Yr_mtx)) < 0
   Yr_mtx = Yr_mtx - min(min(Yr_mtx)) + eps; 
end

if min(min(Yt_mtx)) < 0
   Yt_mtx = Yt_mtx - min(min(Yt_mtx)) + eps; 
end

% TRAINING
% ================================================================
alpha = 1e-6;

C = zeros(max(Class_test_inx)); % initialization
for k = 1:MCRuns
  
    A = rand(size(Yr_mtx,1),J);
    X = rand(J,size(Yr_mtx,2));
    
    tic
    [Ar,Xr,res] = nmf_fast_hals(Yr_mtx,A,X,1e-8,1e-7,MaxIter,1);
    elapsed_time = toc;

    % TESTING
    % ================================================================
    Xt = rand(J,size(Yt_mtx,2));
    Xt = fast_hals(Ar,Yt_mtx,Xt,300);
    
    Xr_tensor = reshape(Xr',[DimYr(1),DimYr(2),J]);
    Xt_tensor = reshape(Xt',[DimYt(1),DimYt(2),J]);
    
    for j = 1:DimYr(2)
        Class_knn_x(:,j) = knnclassify(squeeze(Xt_tensor(:,j,:)),squeeze(Xr_tensor(:,j,:)),Class_train_inx,1,'cosine');
    end
    Class_knn = mode(Class_knn_x,2);
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
