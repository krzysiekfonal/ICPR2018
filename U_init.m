function [Uinit] = U_init(X,R,init)
N = ndims(X);
dimorder = 1:N;
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
            %Uinit{n} = bsxfun(@rdivide,Uinit_unorm,sqrt(sum(Uinit_unorm.^2,1)));
            for j=1:R
                Uinit{n}(:,j) = Uinit_unorm(:,j)/(norm(Uinit_unorm(:,j)) + eps);
            end
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
end
