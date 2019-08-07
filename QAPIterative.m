function [Jhat,Pihat,maxproblem] = QAPIterative(A,B,Niter,epstol)
%QAPIterative Iterative Solver of the Quadratic Assignment Problem
% Use:
%   [Jhat,Pihat,maxproblem] = QAPIterative(A,B,Niter,epstol)
% Inputs:
%  A,B = nxn symmetric matrices
%  Niter : interger == maximum number of iterations
%  epstol : float == stopping criterion
% Outputs:
%  J : real number
%  Pi: n-vector of permutation
%  maxproblem = 1 if the max problem produces the optimal solution; =0 if
%  the min problem yields the optimal permutation
% 12 July 2019, Radu Balan

n = size(A,1);

% Initialize
Pi_c = (1:n)'; % Identity permutation
Jc = trace(A*B);
currdiff = 1e10;
it = 1;
Idn = eye(n);
mxp = 0;

while ((it <=Niter) && (currdiff > epstol))
    % Prepare data
    % Create Permutation matrix
    Phi = Idn(Pi_c,1:n);
    % Call problem
    [Jn,Pin,mxpn] = ABNetwork(A,B*Phi,A,B);

    % Update
    it = it+1;
    currdiff = Jn - Jc;
    Jc = Jn;
    Pi_c = Pin;
    mxp = mxp + mxpn;
end

Jhat = Jn;
Pihat = Pin;
maxproblem = mxp/(it-1);

end

