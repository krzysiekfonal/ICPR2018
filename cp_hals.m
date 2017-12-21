function [P,Uinit,output] = cp_hals(X,R,varargin)

%CP_HALS Compute a CP decomposition of any type of tensor.
%
%   P = CP_HALS(X,R) computes an estimate of the best rank-R
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

%% Extract number of dimensions and norm of X.
N = ndims(X);
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

%% Error checking 

%% Set up and error checking on initial guess for U.
if iscell(init)
    Uinit = init;
    if numel(Uinit) ~= N
        error('OPTS.init does not have %d cells',N);
    end
    for n = dimorder(1:end);
        if ~isequal(size(Uinit{n}),[size(X,n) R])
            error('OPTS.init{%d} is the wrong size',n);
        end
    end
else
    
    if strcmp(init,'random')
        Uinit = cell(N,1);
        for n = dimorder(1:end)
            Uinit_unorm = rand(size(X,n),R);
            Uinit{n} = bsxfun(@rdivide,Uinit_unorm,sqrt(sum(Uinit_unorm.^2,1)));
        end
    elseif strcmp(init,'nvecs') || strcmp(init,'eigs') 
        Uinit = cell(N,1);
        for n = dimorder(1:end)
            Uinit{n} = nvecs(X,n,R);
        end
    else
        error('The selected initialization method is not supported');
    end
end

%% Set up for iterations - initializing U and the fit.
U = Uinit;
fit = 0;

lambda = ones(R,1);
P = ktensor(lambda,U);
normresidual(1) = sqrt( normX^2 + norm(P)^2 - 2 * innerprod(X,P) )/normX;

if printitn>0
  fprintf('\nCP_HALS:\n');
end

%% Main Loop: Iterate until convergence

if (isa(X,'sptensor') || isa(X,'tensor')) && (exist('cpals_core','file') == 3)
 
    %fprintf('Using C++ code\n');
    [lambda,U] = cpals_core(X, Uinit, fitchangetol, maxiters, dimorder);
    P = ktensor(lambda,U);
    
else
    
    % FAST HALS NTF (Algorithm 7.5)
      T1 = ones(R,R);
      for i = 1:N
          T1 = T1 .* (U{i}'*U{i});
      end
                
    for iter = 1:maxiters
        
        fitold = fit;
        gamma = sum(U{N}.^2,1)';
        
        % Iterate over all N modes of the tensor
        for n = dimorder(1:end)
            
            if n == N
                gamma = ones(R,1);
            end
            
            % Calculate Unew = X_(n) * khatrirao(all U except n, 'r').
            T2 = mttkrp(X,U,n);
            
            % Compute T3
            T3 = T1./(U{n}'*U{n});
            
            
            % Loop for j
            for j = 1:R
               % U{n}(:,j) = max(eps,gamma(j)*U{n}(:,j) + T2(:,j) - U{n}*T3(:,j));
                U{n}(:,j) = gamma(j)*U{n}(:,j) + T2(:,j) - U{n}*T3(:,j);
                if n ~= N
                   U{n}(:,j) = U{n}(:,j)/(norm(U{n}(:,j)) + eps);
                end
            end
                         
            T1 = T3.*(U{n}'*U{n});
            
        end % for n
        
        P = ktensor(lambda,U);
        normresidual(iter+1) = sqrt( normX^2 + norm(P)^2 - 2 * innerprod(X,P) )/normX;
        fit = 1 - (normresidual(iter+1)); %fraction explained by model
        fitchange = abs(fitold - fit);
        
        if mod(iter,printitn)==0
            fprintf(' Iter %2d: fit = %e fitdelta = %7.1e\n', iter, fit, fitchange);
        end
        
        % Check for convergence
        if (iter > 1) && (fitchange < fitchangetol)
            break;
        end        
    end   
end


%% Clean up final result
% Arrange the final tensor so that the columns are normalized.
%P = arrange(P);
% Fix the signs
%P = fixsigns(P);

if printitn>0
  normresidual(iter+1) = sqrt( normX^2 + norm(P)^2 - 2 * innerprod(X,P) )/normX;
  fit = 1 - (normresidual(iter+1)); %fraction explained by model
  fprintf(' Final fit = %e \n', fit);
end

output = struct;
output.params = params.Results;
output.iters = iter;
output.normresidual = normresidual;
