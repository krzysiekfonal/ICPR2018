% EMG/MMG classification using 3D tensors (time merged with channels)
% obtained by the STFT

% Import danych
Sx = load([pwd '\dane\osoba_1_full.mat']);
Xraw = Sx.osoba_1;

disp(['Loading passed successfuly']);

% Transformation to 4D tensor of spectrograms
dr = 2; % decimation order
Yemg = preprocessing_STCM(Xraw,1,dr);
Ymmg = preprocessing_STCM(Xraw,2,dr);

Yemg = Yemg(:,:,1:50,:);
Ymmg = Ymmg(:,:,1:50,:);
Yemg = bsxfun(@minus,Yemg,mean(mean(Yemg,1),2));


disp(['Preprocessing passed']);
[I_1,I_2,I_3,I_4] = size(Yemg);

% Settings
NoCV = 5; % number of CV-folds
MCRuns = 10; % MC runs
MaxIter = 20; % number of iterations (iterative methods)
J = 20; % number of latent components
Jt = [20 20 20]; % Rz?dy J_k dla k = 1, ..,3
Jtc = [30 30 30]; % Rz?dy J_k dla k = 1, ..,3
visualization = 1;

ET = zeros(NoCV,6);
delta = zeros(NoCV,6);
C = zeros([I_1 I_1 NoCV, 6]);

% Cross-validtion partition
for n = 1:NoCV
    
    [Yemg_train,Yemg_test,Class_train_inx_emg,Class_test_inx_emg] = CV_function(Yemg,NoCV,n); % cross-validation
    [Ymmg_train,Ymmg_test,Class_train_inx_mmg,Class_test_inx_mmg] = CV_function(Ymmg,NoCV,n); % cross-validation
    
% Algorithms    
% ========================================================================    
    
   % Orth-Tucker
    [C_hosvd,delta_hosvd,elapsed_time_hosvd] =  Tucker_orth_3D(Yemg_train,Yemg_test,Class_train_inx_emg,Class_test_inx_emg,Jt);
    C(:,:,n,1) = C_hosvd; delta(n,1) = delta_hosvd; ET(n,1) = elapsed_time_hosvd;
    disp(['EMG passed']);
    
    [C_hosvd,delta_hosvd,elapsed_time_hosvd] =  Tucker_orth_3D(Ymmg_train,Ymmg_test,Class_train_inx_mmg,Class_test_inx_mmg,Jt);
    C(:,:,n,2) = C_hosvd; delta(n,2) = delta_hosvd; ET(n,2) = elapsed_time_hosvd;
    disp(['MMG passed']);
    
% %     [C_hosvd,delta_hosvd,elapsed_time_hosvd] =  Tucker_orth_3D_mixed(Yemg_train,Yemg_test,Class_train_inx_emg,Class_test_inx_emg,...
% %          Ymmg_train,Ymmg_test,Class_train_inx_mmg,Class_test_inx_mmg,Jt);
%     C(:,:,n,3) = C_hosvd; delta(n,3) = delta_hosvd; ET(n,3) = elapsed_time_hosvd;
%     disp(['Mixted passed']);
%     
%    % Orth-Tucker(core)
%     [C_hosvd,delta_hosvd,elapsed_time_hosvd] =  Tucker_orth_3D_core(Y_train,Y_test,Class_train_inx,Class_test_inx,Jtc);
%     C(:,:,n,5) = C_hosvd; delta(n,5) = delta_hosvd; ET(n,5) = elapsed_time_hosvd;
%         
end

        % Visualization
        if visualization==1
            
            figure
            subplot(1,3,1)
            hintonw((squeeze(mean(C(:,:,:,1),3)))')
            title(['EMG: P = ',num2str( mean(delta(:,1)) ),' %'])
            ylabel('Output')
            subplot(1,3,2)
            hintonw((squeeze(mean(C(:,:,:,2),3)))')
            title(['MMG: P = ',num2str( mean(delta(:,2)) ),' %'])
            ylabel('Output')
            subplot(1,3,3)
            hintonw((squeeze(mean(C(:,:,:,3),3)))')
            title(['Mixed: P = ',num2str( mean(delta(:,3)) ),' %'])
            ylabel('Output')
            set(gcf,'Color',[1 1 1])
            
        end
        
%save '3D_EMG_person_1_10classes' C delta ET NoCV MCRuns MaxIter J Jt Jtc 
    