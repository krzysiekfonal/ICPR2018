function [C,delta,elapsed_time,res] = HALS_CP_4D(Yr,Yt,Class_train_inx,Class_test_inx,J,MCRuns,Tol,MaxIter,init)

% if min(min(min(Yr))) < 0
%    Yr = Yr - min(min(min(Yr))) + eps; 
% end
% 
% if min(min(min(Yt))) < 0
%    Yt = Yt - min(min(min(Yt))) + eps; 
% end
% opts = cp_fLMa;
% opts = cp_init;
% opts.maxiters = MaxIter;
% opts.init = {'nvecs' 'rand'};
% opts.printitn = 1;
% opts.tol = 1e-4;

% TRAINING
% ================================================================
alpha = 1e-6;
C = zeros(max(Class_test_inx)); % initialization
res = [];

for k = 1:MCRuns
  
    tic
    [P,Uinit,output] = cp_hals(tensor(Yr),J,'tol',Tol,'maxiters',MaxIter,'printitn',1,'init',init);
    %[P] = cp_fhals(tensor(Yr),J,opts);
    Ar = P.U{1}; Br = P.U{2}; Cr = P.U{3}; Dr = P.U{4};
    elapsed_time = toc;

    % TESTING
    % ================================================================
%     Y3 = reshape(permute(Yt,[3 1 2]),size(Yt,3),size(Yt,1)*size(Yt,2));
% 
%     At = Ar'*Ar;
%     Bt = Br'*Br;
    
 %   Ct = (Y3*kr(Br,Ar))*inv(Bt.*At + alpha*eye(J));
        
    Dt = hals_4mode(Yt,Ar,Br,Cr,J,100);
       
    Dt = Dt.*repmat(1./sqrt(sum(Dt.^2,2)+eps),1,size(Dt,2));
    Class_knn = knnclassify(Dt,Dr,Class_train_inx,1,'cosine');
    delta = 100*(length(find((Class_knn - Class_test_inx)==0))/length(Class_test_inx));
    
    fprintf('\ndelta: %f \n',delta);

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

res = output.normresidual; 

end
