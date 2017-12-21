function [C,delta,elapsed_time] = Pca_3D(Y_train,Y_test,Class_train_inx,Class_test_inx,J)

%S = load('EMG_tensors_5x200_3D_TC_merged_small');
DimY = size(Y_train);
Yr = reshape(Y_train,[DimY(1)*DimY(2)*DimY(3),DimY(4)]);
Yt = reshape(Y_test,[DimY(1)*DimY(2)*DimY(3),size(Y_test,4)]); 

% TRAINING
% ================================================================

% PCA    
tic
%Yrs = Yr - mean(Yr,2)*ones(1,size(Yr,2));
Yrs = Yr - ones(size(Yr,1),1)*mean(Yr,1);
Cr=(Yrs*Yrs')/(size(Yrs,2));

clear Yr Y_train Y_test

[Vr,Dr]=eigs(Cr,J);
Zr = Vr'*Yrs;
elapsed_time = toc;

% TESTING
% ================================================================
Yts = Yt - ones(size(Yt,1),1)*mean(Yt,1);
%Yts = Yt - mean(Yt,2)*ones(1,size(Yt,2));
Zt = Vr'*Yts;

Class_knn = knnclassify(Zt',Zr',Class_train_inx,1,'cosine');
%Class_knn = knnclassify(Zt',Zr',Class_train_inx,1);
delta = 100*(length(find((Class_knn - Class_test_inx)==0))/length(Class_test_inx));
 
% Macierz prawd
T = zeros(max(Class_test_inx),length(Class_test_inx));
Ts = T;
I = eye(max(Class_test_inx));
for i = 1:length(Class_test_inx)
    T(:,i) = I(:,Class_test_inx(i));
    Ts(:,i) = I(:,Class_knn(i));
end
[C,rate]=confmat(Ts',T');
    
end