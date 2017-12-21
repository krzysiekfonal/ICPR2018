function [C,delta,elapsed_time] = Tucker_orth_4D_core(Yr,Yt,Class_train_inx,Class_test_inx,J)

% % Centralization of each spectrogram
% for i = 1:size(Yr,3)
%     Yr(:,:,i) = Yr(:,:,i) - mean2(Yr(:,:,i));                    
% end
% 
% for i = 1:size(Yt,3)
%     Yt(:,:,i) = Yt(:,:,i) - mean2(Yt(:,:,i));                    
% end

% TRAINING
% ================================================================

tic
% Size
DimY = size(Yr);

% Unfolding
Y1 = reshape(Yr,DimY(1),DimY(2)*DimY(3)*DimY(4));
Y2 = reshape(permute(Yr,[2 1 3 4]),DimY(2),DimY(1)*DimY(3)*DimY(4));
Y3 = reshape(permute(Yr,[3 1 2 4]),DimY(3),DimY(1)*DimY(2)*DimY(4));
Y4 = reshape(permute(Yr,[4 1 2 3]),DimY(4),DimY(1)*DimY(2)*DimY(3));

% Tensor decomposition
[E1,D1] = eig(Y1*Y1');
Ar = fliplr(E1(:,DimY(1)-J(1)+1:DimY(1)));

[E2,D2] = eig(Y2*Y2');
Br = fliplr(E2(:,DimY(2)-J(2)+1:DimY(2)));
        
[E3,D3] = eig(Y3*Y3');
Cr = fliplr(E3(:,DimY(3)-J(3)+1:DimY(3)));
      
[E4,D4] = eig(Y4*Y4');
Dr = fliplr(E4(:,DimY(4)-J(4)+1:DimY(4)));
              
Gr = ntimes(ntimes(ntimes(Yr,Ar',1,2),Br',1,2),Cr',1,2); % core tensor
              
elapsed_time = toc;
    
% TESTING
% ================================================================
Gt = ntimes(ntimes(ntimes(Yt,Ar',1,2),Br',1,2),Cr',1,2); % out 3-mode core tensor for testing data
Gt1 = reshape(Gt,size(Gt,1),size(Gt,2)*size(Gt,3)*size(Gt,4));
Gr1 = reshape(Gr,size(Gr,1),size(Gr,2)*size(Gr,3)*size(Gr,4));

% Klasyfikacja 1-NN
Class_knn = knnclassify(Gt1,Gr1,Class_train_inx,1,'cosine');

% Dok³adnoœæ klasyfikacji
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


