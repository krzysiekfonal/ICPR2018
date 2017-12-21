function [Y_train,Y_test,Class_train_inx,Class_test_inx] = CV_function(Y,NoCV,n)

    [I_1,I_2,I_3,I_4] = size(Y);
    NoTrials = size(Y,2);
    Nx = round(NoTrials/NoCV); % number of samples in each fold

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
    
end