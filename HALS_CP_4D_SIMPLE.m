function [C,delta,elapsed_time,res] = HALS_CP_4D_SIMPLE(Yr,Xr,Yt,Xt,Class_train_inx,Class_test_inx,JC,JI,alpha,MCRuns,Tol,MaxIter,init)

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
C = zeros(max(Class_test_inx)); % initialization
res = [];

for k = 1:MCRuns
  
    tic
    [Cr,Ar,Br,output] = simple_hals(Yr,Xr,JI+JC,JC,Tol,MaxIter);
    CAr = [Cr{4} Ar{4}];
    CBr = [Cr{4} Br{4}];    
    Dr = [ Ar{4} Br{4}];
    elapsed_time = toc;
    
    
    [Ct,At,Bt] = hals_nmode_simple(Yt,Xt,Cr,Ar,Br,JI+JC,JC,4,MaxIter);
    CAt = [Ct{4} At{4}];
    CBt = [Ct{4} Bt{4}];
    Dt = [At{4} Bt{4}];
    
    Dt = Dt.*repmat(1./sqrt(sum(Dt.^2,2)+eps),1,size(Dt,2));    
    Class_knn = knnclassify(Dt,Dr,Class_train_inx,1,'cosine');
    delta = 100*(length(find((Class_knn - Class_test_inx)==0))/length(Class_test_inx));
    
    %CAt = CAt.*repmat(1./sqrt(sum(CAt.^2,2)+eps),1,size(CAt,2));
    %Class_knn = knnclassify(CAt,CAr,Class_train_inx,1,'cosine');
    %deltaA = 100*(length(find((Class_knn - Class_test_inx)==0))/length(Class_test_inx));
    
    %CBt = CBt.*repmat(1./sqrt(sum(CBt.^2,2)+eps),1,size(CBt,2));
    %Class_knn = knnclassify(CBt,CBr,Class_train_inx,1,'cosine');
    %deltaB = 100*(length(find((Class_knn - Class_test_inx)==0))/length(Class_test_inx));
    
    %Ct{4} = Ct{4}.*repmat(1./sqrt(sum(Ct{4}.^2,2)+eps),1,size(Ct{4},2));
    %Class_knn = knnclassify(Ct{4},Cr.U{4},Class_train_inx,1,'cosine');
    %deltaC = 100*(length(find((Class_knn - Class_test_inx)==0))/length(Class_test_inx));
    

    fprintf('\ndelta: %f \n',delta);
    %fprintf('\ndeltaA: %f \n',deltaA);
    %fprintf('\ndeltaB: %f \n',deltaB);
    %fprintf('\ndeltaC: %f \n',deltaC);


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