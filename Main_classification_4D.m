% EMG/MMG classification using 4D tensors obtained by the STFT
%function Main_classification_4D(varargin)

% Data import
Sx = load([pwd '/dane/osoba_1_full.mat']);
Xraw = Sx.osoba_1;

disp(['Loading passed successfuly']);

% Transformation to 5D tensor of spectrograms
channel_select = 1; % (1 - EMG, 2 - MMG)
Y = preprocessing_Spect_5D(Xraw,channel_select);
disp(['Preprocessing passed']);
[I_1,I_2,I_3,I_4,I_5] = size(Y);
% Y: class x trial x channel x frequency x time
feature_merge = 2; % (1 - EMG, 2 - MMG)
X = preprocessing_Spect_5D(Xraw,feature_merge);
disp(['Preprocessing2 passed']);
% [I_1,I_2,I_3,I_4,I_5] = size(X);

% Settings
NoCV = 5; % number of CV-folds
MCRuns = 1; % MC runsfor n = 1:NoCV

MaxIter = 100; % number of iterations (iterative methods)
Tol = 1e-4; % tolerance
J = 20; % number of latent components
JC = 5; % number of common part for feature fusioned algorithms
alpha = 0.5; % weight of importance of first tensor in FF algs
Jt = [10 10 8 20]; % Ranks J_k for k = 1, ..,3
Jtc = [30 30 8 30]; % Ranks J_k for k = 1, ..,3

visualization = 1;

NoTrials = size(Y,2);
Nx = round(NoTrials/NoCV); % number of samples in each fold

ET = zeros(NoCV,6);
delta = zeros(NoCV,6);
C = zeros([I_1 I_1 NoCV, 6]);

% Cross-validtion partition
for n = 1:NoCV
%for n = 1:1    
   
    disp(['Starts CV number: ',num2str(n)]);
    inx_selected = (Nx*(n-1)+1):Nx*n;
    if max(inx_selected) > NoTrials
       inx_test = (Nx*(n-1)+1):NoTrials;
    else            
       inx_test = (Nx*(n-1)+1):Nx*n;
    end
    inx_train = setdiff(1:NoTrials,inx_test);
        
    Y_train_5D = Y(:,inx_train,:,:,:);
    Y_test_5D = Y(:,inx_test,:,:,:);
    
    X_train_5D = X(:,inx_train,:,:,:);
    X_test_5D = X(:,inx_test,:,:,:);

    Ytr_unfolded_1 = reshape(permute(Y_train_5D,[2 1 3 4 5]),[I_1*length(inx_train),I_3 I_4 I_5]);
    Ytest_unfolded_1 = reshape(permute(Y_test_5D,[2 1 3 4 5]),[I_1*length(inx_test),I_3 I_4 I_5]);
    
    Xtr_unfolded_1 = reshape(permute(X_train_5D,[2 1 3 4 5]),[I_1*length(inx_train),I_3 I_4 I_5]);
    Xtest_unfolded_1 = reshape(permute(X_test_5D,[2 1 3 4 5]),[I_1*length(inx_test),I_3 I_4 I_5]);

    Inx_train_class = ones(length(inx_train),I_1)*diag([1:I_1]);
    Class_train_inx = Inx_train_class(:);

    Inx_test_class = ones(length(inx_test),I_1)*diag([1:I_1]);
    Class_test_inx = Inx_test_class(:);

    Y_train = permute(Ytr_unfolded_1,[4 3 2 1]);
    Y_test = permute(Ytest_unfolded_1,[4 3 2 1]);
    
    X_train = permute(Xtr_unfolded_1,[4 3 2 1]);
    X_test = permute(Xtest_unfolded_1,[4 3 2 1]);
  
    % Y: time x frequency x channel x samples (class*trials)
    
% Algorithms    
% ========================================================================    
    
    % PCA
%    [C_pca,delta_pca,elapsed_time_pca] = Pca_4D(Y_train,Y_test,Class_train_inx,Class_test_inx,J);
%     C(:,:,n,1) = C_pca; delta(n,1) = delta_pca; ET(n,1) = elapsed_time_pca;
% 
    % NMF
     [C_cp,delta_cp,elapsed_time_cp,res_nmf] =  NMF_4D(Y_train,Y_test,Class_train_inx,Class_test_inx,J,MCRuns,Tol,MaxIter);
     C(:,:,n,6) = C_cp; delta(n,6) = delta_cp; ET(n,6) = elapsed_time_cp; res.res_nmf = res_nmf;
%         
%    % Nonnegative ALS-CP
     [C_cp,delta_cp,elapsed_time_cp,res_cp_als] =  CP_4D(Y_train,Y_test,Class_train_inx,Class_test_inx,J,MCRuns,Tol,MaxIter);
     C(:,:,n,4) = C_cp; delta(n,4) = delta_cp; ET(n,4) = elapsed_time_cp; res.res_cp_als = res_cp_als;
  
%    Uinit = U_init(Y_train,J,'random');
    % Nonnegative HALS-CP
    [C_cp,delta_cp,elapsed_time_cp,res_cp_hals] =  HALS_CP_4D(Y_train,Y_test,Class_train_inx,Class_test_inx,J,MCRuns,Tol,MaxIter,'random');
    C(:,:,n,1) = C_cp; delta(n,1) = delta_cp; ET(n,1) = elapsed_time_cp; res.res_cp_hals = res_cp_hals;
    
    % HALS-CP FF
    [C_cp,delta_cp_ff,elapsed_time_cp,res_cp_hals_ff] =  HALS_CP_4D_FF(Y_train,X_train,Y_test,X_test,Class_train_inx,Class_test_inx,JC,J-JC,alpha,MCRuns,Tol,MaxIter,'random');
    C(:,:,n,3) = C_cp; delta(n,3) = delta_cp_ff; ET(n,3) = elapsed_time_cp; res.res_cp_hals_ff = res_cp_hals_ff;
    
    % HALS-CP SIMPLE
    [C_cp,delta_cp_simple,elapsed_time_cp,res_cp_hals_simple] =  HALS_CP_4D_SIMPLE(Y_train,X_train,Y_test,X_test,Class_train_inx,Class_test_inx,JC,J-JC,alpha,MCRuns,Tol,MaxIter,'random');
    C(:,:,n,2) = C_cp; delta(n,2) = delta_cp_simple; ET(n,2) = elapsed_time_cp; res.res_cp_hals_simple = res_cp_hals_simple;

   % Orth-Tucker
    [C_hosvd,delta_hosvd,elapsed_time_hosvd] =  Tucker_orth_4D(Y_train,Y_test,Class_train_inx,Class_test_inx,Jt);
    C(:,:,n,5) = C_hosvd; delta(n,5) = delta_hosvd; ET(n,5) = elapsed_time_hosvd;

   % Orth-Tucker(core)
%    [C_hosvd,delta_hosvd,elapsed_time_hosvd] =  Tucker_orth_4D_core(Y_train,Y_test,Class_train_inx,Class_test_inx,Jtc);
%    C(:,:,n,6) = C_hosvd; delta(n,6) = delta_hosvd; ET(n,6) = elapsed_time_hosvd;
% 
%    % Multiclass-NMF
%     [C_cp,delta_cp,elapsed_time_cp] =  Multiclass_NMF_4D(Y_train,Y_test,Class_train_inx,Class_test_inx,J,MCRuns,MaxIter);
%      C(:,:,n,6) = C_cp; delta(n,6) = delta_cp; ET(n,6) = elapsed_time_cp;
      
end


        % Visualization
        if visualization==1
            
            figure
            subplot(3,2,1)
            hintonw((squeeze(mean(C(:,:,:,3),3)))')
            title(['FF-HALS: P = ',num2str( mean(delta(:,3)) ),' %'])
            ylabel('Output')
            
            subplot(3,2,2)
            hintonw((squeeze(mean(C(:,:,:,2),3)))')
            title(['LCPTD-HALS: P = ',num2str( mean(delta(:,2)) ),' %'])
            ylabel('Output')
            
            subplot(3,2,3)
            hintonw((squeeze(mean(C(:,:,:,1),3)))')
            title(['CP-HALS: P = ',num2str( mean(delta(:,1)) ),' %'])
            ylabel('Output')            
            
            subplot(3,2,4)
            hintonw((squeeze(mean(C(:,:,:,4),3)))')            
            title(['CP-ALS: P = ',num2str( mean(delta(:,4)) ),' %'])
            ylabel('Output')            
            
            subplot(3,2,5)
            hintonw((squeeze(mean(C(:,:,:,5),3)))')
            title(['HO-SVD: P = ',num2str( mean(delta(:,5)) ),' %'])
            ylabel('Output')
            
            subplot(3,2,6)            
            hintonw((squeeze(mean(C(:,:,:,6),3)))')
            title(['NMF: P = ',num2str( mean(delta(:,6)) ), ' %'])
            ylabel('Output')
            
            set(gcf,'Color',[1 1 1])
            
        end
 
%save '4D_EMG_person_1_10classes' C delta ET NoCV MCRuns MaxIter J Jt Jtc res

%end
    