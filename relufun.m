function V=relufun(U)
% relufun REctified Linear Unit FUNction
% Use:
%  V=relufun(U)
% Input:
%  U is a NxM matrix
% Outputs:
%  V is a NxM matrix , V(i,j) = U(i,j) if U(i,j)>=0 , =0 otherwise
% 
% 14 July 2019, Radu Balan

N = size(U,1);
M = size(U,2);
Uarray(1,1:N,1:M) = U;
Uarray(2,1:N,1:M) = zeros(N,M);

V = reshape(max(Uarray),N,M);

end

