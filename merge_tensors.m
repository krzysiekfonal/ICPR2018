function [ CI ] = merge_tensors( C, I )
%MERGE_TENSORS Merges 2 tensors: C and I
%   C is the tensor consisting of Common part, I is a tensor consisting of 
%   individual part

n = ndims(C);
U = cell(n, 1);
for i = 1:n
    U{i} = [C.U{i} I.U{i}];
end

CI = ktensor(ones(size(U{1},2),1), U);

end

