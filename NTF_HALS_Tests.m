clear

benchmark = 1; % 1- rand, 2 - randn+, 3 - MNIST, 4 - ORL
J = 20; % rank of factorizations
JC = 4; % number of common rank
alpha = 0.5; % factor of importance of Y tensor over X tensor
JI = J-JC;

switch benchmark 
    
    case 1 % rand
        
        I = [50 50 500]; % dimensions in each mode
        N = length(I); % number of modes

        UC = cell([N,1]);
        UA = cell([N,1]);
        UB = cell([N,1]);
        for n = 1:N
            UC{n} = rand(I(n),JC); % factor for the n-th mode
            UA{n} = rand(I(n),JI); % factor for the n-th mode
            UB{n} = rand(I(n),JI); % factor for the n-th mode
            UCA{n} = [UC{n} UA{n}]; % merged C and A factor
            UCB{n} = [UC{n} UB{n}]; % merged C and B factor
        end
        lambda = ones(J,1); % super-diagonal from the core tensor
        Y = ktensor(lambda,UCA); % data tensor Y
        X = ktensor(lambda,UCB); % data tensor X
        
     case 2 % randn
        
        I = [50 50 500]; % dimensions in each mode
        N = length(I); % number of modes

        U = cell([N,1]);
        for n = 1:N
            U{n} = max(0,randn(I(n),J)); % factor for the n-th mode
        end
        lambda = ones(J,1); % super-diagonal from the core tensor
        Y = ktensor(lambda,U); % data tensor    
        
end % switch


% Parameters
Tol = 1e-5;
MaxIter = 100;
Show_inx = 1;
alg = 2; % 1 - hals-ff, 2 - hals-lcptd

%% Algorithms
tic
if alg==1
    [tCU,tAU,tBU,output] = cp_hals_ff(Y,X,JC,JI,alpha,'tol',Tol,'maxiters',MaxIter,'printitn',Show_inx);
else
    [tCU,tAU,tBU,output] = simple_hals(double(Y),double(X),J,JC,Tol,MaxIter);
end
%[Yhat,Uhat,Uinit,output] = cp_hals(tensor(Y),J,'tol',Tol,'maxiters',MaxIter,'printitn',Show_inx);
elapsed_time = toc;    
res = output.normresidual;
    

%% Validation
validation_ntf(UC,UA,UB,tCU,tAU,tBU,res,elapsed_time);
