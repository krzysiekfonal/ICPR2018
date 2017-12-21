function [C,delta,elapsed_time] = Tucker_orth_3D(Yr,Yt,Class_train_inx,Class_test_inx,J)

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

% Size
DimY = size(Yr);

tic
% Unfolding
Y1 = reshape(Yr,DimY(1),DimY(2)*DimY(3));
Y2 = reshape(permute(Yr,[2 1 3]),DimY(2),DimY(1)*DimY(3));
Y3 = reshape(permute(Yr,[3 1 2]),DimY(3),DimY(1)*DimY(2));

% Tensor decomposition
[Ar,D1] = eigs(Y1*Y1',J(1));
[Br,D2] = eigs(Y2*Y2',J(2));
[Cr,D3] = eigs(Y3*Y3',J(3));
              
Gr = ntimes(ntimes(ntimes(Yr,Ar',1,2),Br',1,2),Cr',1,2); % core tensor
Cr = Cr.*repmat(1./sqrt(sum(Cr.^2,2)+eps),1,size(Cr,2));

elapsed_time = toc;
    
% TESTING
% ================================================================
Y3 = reshape(permute(Yt,[3 1 2]),size(Yt,3),size(Yt,1)*size(Yt,2));
G3 = reshape(permute(Gr,[3 1 2]),[J(3),J(1)*J(2)]);
   
Ct = Y3*pinv(double(G3)*(kron(Br,Ar))');
Ct = Ct.*repmat(1./sqrt(sum(Ct.^2,2)+eps),1,size(Ct,2));

% Klasyfikacja 1-NN
Class_knn = knnclassify(Ct,Cr,Class_train_inx,1,'cosine');

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


