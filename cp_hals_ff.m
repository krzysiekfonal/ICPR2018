function [UC,UA,UB,output] = cp_hals_ff(Y,X,RC,RI,alpha,varargin)

%CP_HALS Compute a CP decomposition of any type of tensor.
%
%   P = CP_HALS_FF(Y,X,R) computes an estimate of the best rank-R
%   CP model of a tensor X using an alternating hierarchical least-squares
%   algorithm.  The input X can be a tensor, sptensor, ktensor, or
%   ttensor. The result P is a ktensor.
%
%   P = CP_HALS(X,R,'param',value,...) specifies optional parameters and
%   values. Valid parameters and their default values are:
%      'tol' - Tolerance on difference in fit {1.0e-4}
%      'maxiters' - Maximum number of iterations {50}
%      'dimorder' - Order to loop through dimensions {1:ndims(A)}
%      'init' - Initial guess [{'random'}|'nvecs'|cell array]
%      'printitn' - Print fit every n iterations; 0 for no printing {1}
%
%   [P,U0] = CP_HALS(...) also returns the initial guess.
%
%   [P,U0,out] = CP_HALS(...) also returns additional output that contains
%   the input parameters.
%
%   Note: The "fit" is defined as 1 - norm(X-full(P))/norm(X) and is
%   loosely the proportion of the data described by the CP model, i.e., a
%   fit of 1 is perfect.
%

%% Extract number of dimensions and norm of X and Y.
dim = size(Y);
N = ndims(X);
normY = norm(Y);
normX = norm(X);

%% Set algorithm parameters from input or by using defaults
params = inputParser;
params.addParamValue('tol',1e-4,@isscalar);
params.addParamValue('maxiters',50,@(x) isscalar(x) & x > 0);
params.addParamValue('dimorder',1:N,@(x) isequal(sort(x),1:N));
params.addParamValue('init', 'random', @(x) (iscell(x) || ismember(x,{'random','nvecs'})));
params.addParamValue('printitn',1,@isscalar);
params.parse(varargin{:});

%% Copy from params object
fitchangetol = params.Results.tol;
maxiters = params.Results.maxiters;
dimorder = params.Results.dimorder;
init = params.Results.init;
printitn = params.Results.printitn;


%% Set up for iterations - initializing U and the fit.
beta = 1 - alpha;
UCinit = U_init(X,RC,init);
UAinit = U_init(Y,RI,init);
UBinit = U_init(X,RI,init);
UC = UCinit; 
UA = UAinit; 
UB = UBinit;
%Uinit = U_init(Y,RC+RI,init);
%UC = cell(N,1);UA = cell(N,1);%UB = cell(N,1);
%for n=1:N
%    UC{n} = Uinit{n}(:,1:RC);
%    UA{n} = Uinit{n}(:,RC+1:RC+RI);
%end
%UB = UA;

fitY = 0; fitX = 0;

lambdaC = ones(RC,1);
lambdaA = ones(RI,1);
lambdaB = ones(RI,1);
C = ktensor(lambdaC,UC);
A = ktensor(lambdaA,UA);
B = ktensor(lambdaB,UB);
CA = merge_tensors(C, A);
CB = merge_tensors(C, B);
normresidualY(1) = sqrt( normY^2 + norm(CA)^2 - 2 * innerprod(Y,CA) )/normY;
normresidualX(1) = sqrt( normX^2 + norm(CB)^2 - 2 * innerprod(X,CB) )/normX;

if printitn>0
  fprintf('\nCP_HALS:\n');
end

%% Main Loop: Iterate until convergence

% FAST HALS NTF (Algorithm 7.5)
T1C = ones(RC,RC);
T1A = ones(RI,RI);
T1B = ones(RI,RI);
for i = 1:N
    T1C = T1C .* (UC{i}'*UC{i});
    T1A = T1A .* (UA{i}'*UA{i});
    T1B = T1B .* (UB{i}'*UB{i});
end

% Calculate unfolded X and Y
Xn = cell(N); Yn = cell(N);
for n = 1:N
    Yn{n} = reshape(permute(double(Y),[n [1:n-1,n+1:N]]),dim(n),prod(dim)/dim(n)); 
    Xn{n} = reshape(permute(double(X),[n [1:n-1,n+1:N]]),dim(n),prod(dim)/dim(n)); 
end

for iter = 1:maxiters

    fitoldY = fitY; fitoldX = fitX;
    gammaC = sum(UC{N}.^2,1)';
    gammaA = sum(UA{N}.^2,1)';
    gammaB = sum(UB{N}.^2,1)';

    % Iterate over all N modes of the tensor
    for n = dimorder(1:end)

        if n == N
            gammaC = ones(RC,1);
            gammaA = ones(RI,1);
            gammaB = ones(RI,1);
        end

        % Calculate khatrirao(all U except n) for U->C,A,B
        krUCexcn = kr_minus_n_new(UC,n);
        krUAexcn = kr_minus_n_new(UA,n);
        krUBexcn = kr_minus_n_new(UB,n);
        
        % Calculate Yd and Xd as a difference between Y,X and individual
        % parts
        Yd = Yn{n} - UA{n} * krUAexcn';
        Xd = Xn{n} - UB{n} * krUBexcn';
                        
        % Calculate Unew = X_(n) * khatrirao(all U except n, 'r').
        T2C = (alpha .* Yd + beta .* Xd) * krUCexcn;
        
        % Compute T3
        T3C = T1C./(UC{n}'*UC{n});
        T3A = T1A./(UA{n}'*UA{n});
        T3B = T1B./(UB{n}'*UB{n});

        % Loop for j for common part
        for j = 1:RC
            UC{n}(:,j) = gammaC(j)*UC{n}(:,j) + T2C(:,j) - UC{n}*T3C(:,j);
            if n ~= N
               UC{n}(:,j) = UC{n}(:,j)/(norm(UC{n}(:,j)) + eps);
            end
        end
        
        % Calculate Yg and Xg as a difference between Y,X and common part
        Yg = Yn{n} - UC{n} * krUCexcn';
        Xg = Xn{n} - UC{n} * krUCexcn';
        
        % Calculate Unew = X_(n) * khatrirao(all U except n, 'r').        
        T2A = Yg * krUAexcn;
        T2B = Xg * krUBexcn;
        
        % Loop for j for individual part
        for j = 1:RI           
            UA{n}(:,j) = gammaA(j)*UA{n}(:,j) + T2A(:,j) - UA{n}*T3A(:,j);
            UB{n}(:,j) = gammaB(j)*UB{n}(:,j) + T2B(:,j) - UB{n}*T3B(:,j);
            if n ~= N
               UA{n}(:,j) = UA{n}(:,j)/(norm(UA{n}(:,j)) + eps);
               UB{n}(:,j) = UB{n}(:,j)/(norm(UB{n}(:,j)) + eps);
            end
        end

        T1C = T3C.*(UC{n}'*UC{n});
        T1A = T3A.*(UA{n}'*UA{n});
        T1B = T3B.*(UB{n}'*UB{n});

    end % for n

    C = ktensor(lambdaC,UC);
    A = ktensor(lambdaA,UA);
    B = ktensor(lambdaB,UB);
    CA = merge_tensors(C, A);
    CB = merge_tensors(C, B);
    normresidualY(iter+1) = sqrt( normY^2 + norm(CA)^2 - 2 * innerprod(Y,CA) )/normY;
    normresidualX(iter+1) = sqrt( normX^2 + norm(CB)^2 - 2 * innerprod(X,CB) )/normX;
    fitY = 1 - (normresidualY(iter+1)); %fraction explained by model
    fitX = 1 - (normresidualX(iter+1)); %fraction explained by model
    fitchangeY = fitoldY - fitY;
    fitchangeX = fitoldX - fitX;

    if mod(iter,printitn)==0
        fprintf(' Iter %2d: fitY = %e fitdeltaY = %7.1e\n', iter, fitY, fitchangeY);
        fprintf(' Iter %2d: fitX = %e fitdeltaX = %7.1e\n', iter, fitX, fitchangeX);
    end
    
    if (fitchangeY > 0 || fitchangeX > 0)
        fprintf(' WARNING: non-negative fitchange');
    end

    % Check for convergence
    if (iter > 1) && (max(abs(fitchangeY),abs(fitchangeX)) < fitchangetol)
        break;
    end        
end   

%% Clean up final result
% Arrange the final tensor so that the columns are normalized.
%P = arrange(P);
% Fix the signs
%P = fixsigns(P);

if printitn>0
  fprintf(' Final fitY = %e \n', fitY);
  fprintf(' Final fitX = %e \n', fitX);
end

output = struct;
output.params = params.Results;
output.iters = iter;
output.normresidualY = normresidualY;
output.normresidualX = normresidualX;
output.normresidual = (normresidualY + normresidualX) ./ 2;

end


