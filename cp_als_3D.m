function [A,B,C,res] = cp_als_3D(Y,R,varargin)

% Size
DimY = size(Y);
normY = sqrt(sum(sum(sum(Y.^2))));
res = [];

%% Set algorithm parameters from input or by using defaults
params = inputParser;
params.addParamValue('tol',1e-4,@isscalar);
params.addParamValue('maxiters',50,@(x) isscalar(x) & x > 0);
params.addParamValue('init', 'random', @(x) (iscell(x) || ismember(x,{'random','nvecs'})));
params.addParamValue('printitn',1,@isscalar);
params.parse(varargin{:});

%% Copy from params object
fitchangetol = params.Results.tol;
maxiters = params.Results.maxiters;
init = params.Results.init;
printitn = params.Results.printitn;

% Initialization
A = rand(DimY(1),R);
B = rand(DimY(2),R);
C = rand(DimY(3),R);
Ir = neye([R R R]);

% Initial residual
Y_hat = ntimes(ntimes(ntimes(Ir,A,1,2),B,1,2),C,1,2); % tensor 3-way
res(1) = sqrt(sum(sum(sum((Y - Y_hat).^2))))/normY;

% Unfolding
Y1 = reshape(Y,DimY(1),DimY(2)*DimY(3));
Y2 = reshape(permute(Y,[2 1 3]),DimY(2),DimY(1)*DimY(3));
Y3 = reshape(permute(Y,[3 1 2]),DimY(3),DimY(1)*DimY(2));

Bt = B'*B;
Ct = C'*C;
fit = 0;

for k = 1:maxiters
        
        fitold = fit;    
    
%        alpha = eps+exp(-k); % damping
        alpha = 1e2./2.^k;
        A = (Y1*kr(C,B))*inv(Ct.*Bt + alpha*eye(R));
        A = A.*repmat(1./sqrt(sum(A.^2,1)+eps),size(A,1),1);
        At = A'*A;
        
        B = (Y2*kr(C,A))*inv(Ct.*At + alpha*eye(R));
        B = B.*repmat(1./sqrt(sum(B.^2,1)+eps),size(B,1),1);
        Bt = B'*B;
        
        C =(Y3*kr(B,A))*inv(Bt.*At + alpha*eye(R));
  %      C = C.*repmat(1./sqrt(sum(C.^2,1)+eps),size(C,1),1);
        Ct = C'*C;
        
        Y_hat = ntimes(ntimes(ntimes(Ir,A,1,2),B,1,2),C,1,2); % tensor 3-way
        res(k+1) = sqrt(sum(sum(sum((Y - Y_hat).^2))))/normY;
        
        fit = 1 - (res(k+1)); %fraction explained by model
        fitchange = abs(fitold - fit);
        
        if mod(k,printitn)==0
            fprintf('CP(ALS): Iter %2d: fit = %e fitdelta = %7.1e\n', k, fit, fitchange);
        end
        
        % Check for convergence
        if (k > 1) && (fitchange < fitchangetol)
            break;
        end        
            
        
end

%  A = A.*repmat(1./sqrt(sum(A.^2,1)+eps),size(A,1),1);
%  B = B.*repmat(1./sqrt(sum(B.^2,1)+eps),size(B,1),1);
%  C = C.*repmat(1./sqrt(sum(C.^2,1)+eps),size(C,1),1);

