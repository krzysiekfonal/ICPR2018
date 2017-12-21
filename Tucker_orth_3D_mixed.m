function [C,delta,elapsed_time] = Tucker_orth_3D_mixed(Yr_emg,Yt_emg,Class_train_inx_emg,Class_test_inx_emg,Yr_mmg,Yt_mmg,Class_train_inx_mmg,Class_test_inx_mmg,J)

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
DimY = size(Yr_emg);

tic
% Unfolding
Y1_emg = reshape(Yr_emg,DimY(1),DimY(2)*DimY(3));
Y2_emg = reshape(permute(Yr_emg,[2 1 3]),DimY(2),DimY(1)*DimY(3));
Y3_emg = reshape(permute(Yr_emg,[3 1 2]),DimY(3),DimY(1)*DimY(2));

Y1_mmg = reshape(Yr_mmg,DimY(1),DimY(2)*DimY(3));
Y2_mmg = reshape(permute(Yr_mmg,[2 1 3]),DimY(2),DimY(1)*DimY(3));
Y3_mmg = reshape(permute(Yr_mmg,[3 1 2]),DimY(3),DimY(1)*DimY(2));

% Tensor decomposition
[Ar_emg,D1_emg] = eigs(Y1_emg*Y1_emg',J(1));
[Br_emg,D2_emg] = eigs(Y2_emg*Y2_emg',J(2));

[Ar_mmg,D1_mmg] = eigs(Y1_mmg*Y1_mmg',J(1));
[Br_mmg,D2_mmg] = eigs(Y2_mmg*Y2_mmg',J(2));

%[Cr_emg,D3_emg] = eigs(Y3_emg*Y3_emg',J(3));
              
Zr = permute(ntimes(ntimes(Yr_emg,(Ar_emg*Ar_emg'),1,2),(Br_emg*Br_emg'),1,2),[3 2 1]); 
Tr = permute(ntimes(ntimes(Yr_mmg,(Ar_mmg*Ar_mmg'),1,2),(Br_mmg*Br_mmg'),1,2),[3 2 1]); 

Z3 = reshape(permute(Zr,[3 1 2]),size(Zr,3),size(Zr,1)*size(Zr,2));
T3 = reshape(permute(Tr,[3 1 2]),size(Tr,3),size(Tr,1)*size(Tr,2));
W = Z3*Z3' + T3*T3' - Y3_emg*Z3' - Z3*Y3_emg' - Y3_mmg*T3' - T3*Y3_mmg';

[Cr,D3] = eigs(W,J(3));
%Cr = Cr.*repmat(1./sqrt(sum(Cr.^2,2)+eps),1,size(Cr,2));
elapsed_time = toc;
    

for ki =1:5
    
    ki
% update Ar
Mr_emg = ntimes(ntimes(Yr_emg,(Br_emg*Br_emg'),2,2),Cr*Cr',2,2);
Mr_mmg = ntimes(ntimes(Yr_mmg,(Br_mmg*Br_mmg'),2,2),Cr*Cr',2,2);


M1_emg = reshape(permute(Mr_emg,[1 2 3]),size(Mr_emg,1),size(Mr_emg,2)*size(Mr_emg,3));
M1_mmg = reshape(permute(Mr_mmg,[1 2 3]),size(Mr_mmg,1),size(Mr_mmg,2)*size(Mr_mmg,3));

Wa_emg = M1_emg*M1_emg' - Y1_emg*M1_emg' - M1_emg*Y1_emg';
Wa_mmg = M1_mmg*M1_mmg' - Y1_mmg*M1_mmg' - M1_mmg*Y1_mmg';

[Ar_emg,D1_emg] = eigs(Wa_emg,J(1));
[Ar_mmg,D2_mmg] = eigs(Wa_mmg,J(1));

% % update B
Mr_emg = ntimes(ntimes(Yr_emg,(Ar_emg*Ar_emg'),1,2),Cr*Cr',2,2);
Mr_mmg = ntimes(ntimes(Yr_mmg,(Ar_mmg*Ar_mmg'),1,2),Cr*Cr',2,2);

M2_emg = reshape(Mr_emg,size(Mr_emg,1),[]);
M2_mmg = reshape(Mr_mmg,size(Mr_mmg,1),[]);

Wa_emg = M2_emg*M2_emg' - Y2_emg*M2_emg' - M2_emg*Y2_emg';
Wa_mmg = M2_mmg*M2_mmg' - Y2_mmg*M2_mmg' - M2_mmg*Y2_mmg';

[Br_emg,D1_emg] = eigs(Wa_emg,J(2));
[Br_mmg,D2_mmg] = eigs(Wa_mmg,J(2));

% update for C

Zr = permute(ntimes(ntimes(Yr_emg,(Ar_emg*Ar_emg'),1,2),(Br_emg*Br_emg'),1,2),[3 2 1]); 
Tr = permute(ntimes(ntimes(Yr_mmg,(Ar_mmg*Ar_mmg'),1,2),(Br_mmg*Br_mmg'),1,2),[3 2 1]); 

Z3 = reshape(permute(Zr,[3 1 2]),size(Zr,3),size(Zr,1)*size(Zr,2));
T3 = reshape(permute(Tr,[3 1 2]),size(Tr,3),size(Tr,1)*size(Tr,2));
W = Z3*Z3' + T3*T3' - Y3_emg*Z3' - Z3*Y3_emg' - Y3_mmg*T3' - T3*Y3_mmg';

[Cr,D3] = eigs(W,J(3));
%Cr = Cr.*repmat(1./sqrt(sum(Cr.^2,2)+eps),1,size(Cr,2));

end

Cr = Cr.*repmat(1./sqrt(sum(Cr.^2,2)+eps),1,size(Cr,2));

% TESTING
% ================================================================
Gr = ntimes(ntimes(ntimes(Yr_emg,Ar_emg',1,2),Br_emg',1,2),Cr',1,2); % core tensor from EMG and common factor C
G3 = reshape(permute(Gr,[3 1 2]),[J(3),J(1)*J(2)]);
Y3 = reshape(permute(Yt_emg,[3 1 2]),size(Yt_emg,3),size(Yt_emg,1)*size(Yt_emg,2));
   
Ct = Y3*pinv(double(G3)*(kron(Br_emg,Ar_emg))');
Ct = Ct.*repmat(1./sqrt(sum(Ct.^2,2)+eps),1,size(Ct,2));

% Klasyfikacja 1-NN
Class_knn = knnclassify(Ct,Cr,Class_train_inx_emg,1,'cosine');

% Dok³adnoœæ klasyfikacji
delta = 100*(length(find((Class_knn - Class_test_inx_emg)==0))/length(Class_test_inx_emg));

% Macierz prawd
T = zeros(max(Class_test_inx_emg),length(Class_test_inx_emg));
Ts = T;
I = eye(max(Class_test_inx_emg));
for i = 1:length(Class_test_inx_emg)
    T(:,i) = I(:,Class_test_inx_emg(i));
    Ts(:,i) = I(:,Class_knn(i));
end
[C,rate]=confmat(Ts',T');

end


