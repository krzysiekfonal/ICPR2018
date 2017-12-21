function [C,delta,elapsed_time] = Tucker_orth_4D(Yr,Yt,Class_train_inx,Class_test_inx,J)

% % Centralization of each spectrogram
% for i = 1:size(Yr,3)
%     for j = 1:size(Yr,4)
%         Yr(:,:,i,j) = Yr(:,:,i,j) - mean2(Yr(:,:,i,j));                    
%     end
% end
% 
% for i = 1:size(Yt,3)
%     for j = 1:size(Yt,4)
%         Yt(:,:,i,j) = Yt(:,:,i,j) - mean2(Yt(:,:,i,j));                    
%     end
% end


% TRAINING
% ================================================================
tic
[Ar,Br,Cr,Dr,Gr] = tucker_hosvd_4D(Yr,J);
elapsed_time = toc;
    
% TESTING
% ================================================================
Y4 = reshape(permute(Yt,[4 1 2 3]),size(Yt,4),size(Yt,1)*size(Yt,2)*size(Yt,3));
G4 = reshape(permute(Gr,[4 1 2 3]),[J(4),J(1)*J(2)*J(3)]);
   
Dt = Y4*pinv(double(G4)*(kron(Cr,kron(Br,Ar)))');
Dt = Dt.*repmat(1./sqrt(sum(Dt.^2,2)+eps),1,size(Dt,2));

% Klasyfikacja 1-NN
Class_knn = knnclassify(Dt,Dr,Class_train_inx,1,'cosine');

% Dok³adnoœæ klasyfikacji
delta = 100*(length(find((Class_knn - Class_test_inx)==0))/length(Class_test_inx));

% Macierz prawd
C = zeros(max(Class_test_inx)); % initialization
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


