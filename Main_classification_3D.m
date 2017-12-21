% EMG/MMG classification using 3D tensors (time merged with channels)
% obtained by the STFT

% Import danych
Sx = load([pwd '\dane\osoba_1_full.mat']);
Xraw = Sx.osoba_1;

% Sx = load([pwd '\dane\osoba_4_full.mat']);
% Xraw = Sx.osoba_4;
% 
% Sx = load([pwd '\dane\5x200.mat']);
% Xraw = Sx.osoba_3_ok;

disp(['Loading passed successfuly']);

% Transformation to 4D tensor of spectrograms
dr = 2; % decimation order
channel_select = 1; % (1 - EMG, 2 - MMG)
Y = preprocessing_STCM(Xraw,channel_select,dr);
disp(['Preprocessing passed']);
[I_1,I_2,I_3,I_4] = size(Y);

% Settings
NoCV = 5; % number of CV-folds
MCRuns = 1; % MC runs
MaxIter = 100; % number of iterations (iterative methods)
Tol = 1e-5; % tolerance
J = 20; % number of latent components
Jt = [20 20 20]; % Rzêdy J_k dla k = 1, ..,3
Jtc = [30 30 30]; % Rzêdy J_k dla k = 1, ..,3
visualization = 1;

NoTrials = size(Y,2);
Nx = round(NoTrials/NoCV); % number of samples in each fold

ET = zeros(NoCV,6);
delta = zeros(NoCV,6);
C = zeros([I_1 I_1 NoCV, 6]);

% Cross-validtion partition
for n = 1:NoCV
    
    disp(['Starts CV number: ',num2str(n)]);
    inx_selected = (Nx*(n-1)+1):Nx*n;
    if max(inx_selected) > NoTrials
       inx_test = (Nx*(n-1)+1):NoTrials;
    else            
       inx_test = (Nx*(n-1)+1):Nx*n;
    end
    inx_train = setdiff(1:NoTrials,inx_test);
        
    Y_train_4D = Y(:,inx_train,:,:,:);
    Y_test_4D = Y(:,inx_test,:,:,:);

    Ytr_unfolded_1 = reshape(permute(Y_train_4D,[2 1 3 4]),[I_1*length(inx_train),I_3 I_4]);
    Ytest_unfolded_1 = reshape(permute(Y_test_4D,[2 1 3 4]),[I_1*length(inx_test),I_3 I_4]);

    Inx_train_class = ones(length(inx_train),I_1)*diag([1:I_1]);
    Class_train_inx = Inx_train_class(:);

    Inx_test_class = ones(length(inx_test),I_1)*diag([1:I_1]);
    Class_test_inx = Inx_test_class(:);

    Y_train = permute(Ytr_unfolded_1,[3 2 1]);
    Y_test = permute(Ytest_unfolded_1,[3 2 1]);
     
% Algorithms    
% ========================================================================    
%     % PCA
%     [C_pca,delta_pca,elapsed_time_pca] = Pca_3D(Y_train,Y_test,Class_train_inx,Class_test_inx,J);
%      C(:,:,n,1) = C_pca; delta(n,1) = delta_pca; ET(n,1) = elapsed_time_pca;

   % NMF
    [C_cp,delta_cp,elapsed_time_cp,res_nmf] =  NMF_3D(Y_train,Y_test,Class_train_inx,Class_test_inx,J,MCRuns,Tol,MaxIter);
     C(:,:,n,2) = C_cp; delta(n,2) = delta_cp; ET(n,2) = elapsed_time_cp; res.res_nmf = res_nmf;
 
%   Nonnegative ALS-CP
% %   [C_cp,delta_cp,elapsed_time_cp] =  CPN_3D(Y_train,Y_test,Class_train_inx,Class_test_inx,J,MCRuns,MaxIter);
     [C_cp,delta_cp,elapsed_time_cp,res_cp_als] =  CP_3D(Y_train,Y_test,Class_train_inx,Class_test_inx,J,MCRuns,Tol,MaxIter);
      C(:,:,n,3) = C_cp; delta(n,3) = delta_cp; ET(n,3) = elapsed_time_cp; res.res_cp_als = res_cp_als;
     
   % Nonnegative HALS-CP
    [C_cp,delta_cp,elapsed_time_cp,res_cp_hals] =  HALS_CP_3D(Y_train,Y_test,Class_train_inx,Class_test_inx,J,MCRuns,Tol,MaxIter);
     C(:,:,n,4) = C_cp; delta(n,4) = delta_cp; ET(n,4) = elapsed_time_cp; res.res_cp_hals = res_cp_hals;
    
   % Orth-Tucker
    [C_hosvd,delta_hosvd,elapsed_time_hosvd] =  Tucker_orth_3D(Y_train,Y_test,Class_train_inx,Class_test_inx,Jt);
    C(:,:,n,5) = C_hosvd; delta(n,5) = delta_hosvd; ET(n,5) = elapsed_time_hosvd;

   % Orth-Tucker(core)
    [C_hosvd,delta_hosvd,elapsed_time_hosvd] =  Tucker_orth_3D_core(Y_train,Y_test,Class_train_inx,Class_test_inx,Jtc);
    C(:,:,n,6) = C_hosvd; delta(n,6) = delta_hosvd; ET(n,6) = elapsed_time_hosvd;
        
end


        % Visualization
        if visualization==1
            
            figure
            subplot(2,3,1)
            hintonw((squeeze(mean(C(:,:,:,1),3)))')
            title(['PCA: P = ',num2str( mean(delta(:,1)) ),' %'])
            ylabel('Output')
            subplot(2,3,2)
            hintonw((squeeze(mean(C(:,:,:,2),3)))')
            title(['NMF: P = ',num2str( mean(delta(:,2)) ),' %'])
            ylabel('Output')
            subplot(2,3,3)
            hintonw((squeeze(mean(C(:,:,:,3),3)))')
            title(['CP(ALS): P = ',num2str( mean(delta(:,3)) ),' %'])
            ylabel('Output')
            subplot(2,3,4)
            hintonw((squeeze(mean(C(:,:,:,4),3)))')
            title(['CP(HALS): P = ',num2str( mean(delta(:,4)) ),' %'])
            ylabel('Output')
            subplot(2,3,5)
            hintonw((squeeze(mean(C(:,:,:,5),3)))')
            title(['HO-SVD(1): P = ',num2str( mean(delta(:,5)) ),' %'])
            ylabel('Output')
            subplot(2,3,6)
            hintonw((squeeze(mean(C(:,:,:,6),3)))')
            title(['HO-SVD(2): P = ',num2str( mean(delta(:,6)) ),' %'])
            ylabel('Output')
            set(gcf,'Color',[1 1 1])
            
        end
        
% save '3D_EMG_person_1_10classes' C delta ET NoCV MCRuns MaxIter J Jt Jtc res