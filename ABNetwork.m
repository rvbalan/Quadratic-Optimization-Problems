function [J,Pi,maxproblem] = ABNetwork(U,V,A,B)
%ABNetwork Quadratic Optimization Function: max trace(Pi*A*Pi'*B)
%   Use:
% [J,Pi,maxproblem] = ABNetwork(U,V,A,B)
% Inputs: 
%  U,V = nxr matrices
%  A,B = nxn symmetric matrices
% Outputs:
%  J : real number
%  Pi: n-vector of permutation
%  maxproblem = 1 if the max problem produces the optimal solution; =0 if
%  the min problem yields the optimal permutation
% Details:
%  Two problems are solves: [J1,Pi1] = max_Pi trace(Pi*U*V') , [J2,Pi2] =
%  min_Pi trace(Pi*U*V'). Then [J,Pi] = max(trace(Pi1*A*P1'*B) , trace(Pi2*A*Pi2'*B))
% 12 July 2019, Radu Balan

n = size(U,1);
C = U*V'; % Matrix to be used in the linear assignment problem: max tr(Pi*C)
% Scale entries to have inf-norm 1
cmax = max(max(abs(C)));
C = C/cmax;

cv = reshape(C',n*n,1); % Vectorize C
% problem: max <w,cv> s.t. w>=0 and L*w=1, for 2n x n*n matrix L (double
% stochasticity condition)
% Create constraint matrix
L = zeros(2*n,n*n);
for k=1:n
    L(k,(k-1)*n+1:k*n)=1;
    L(n+k,k:n:n*n)=1;
end

% Solve the maximum
problem.f = -cv;
problem.Aineq = -eye(n*n);
problem.bineq = zeros(n*n,1);
problem.Aeq = L;
problem.beq = ones(2*n,1);
problem.lb = [];
problem.ub =[];
problem.solver = 'linprog';
problem.options = optimoptions('linprog','Display','none');
%problem.options = optimoptions('OptimalityTolerance',
%[w,~,~,~,~] = linprog(-cv,-eye(n*n),zeros(n*n,1),L,ones(2*n,1));
[w,~,~,~,~] = linprog(problem);
% reshape w
W = reshape(w,n,n);
% Turn it into a 0/1 matrix
[~,Pi1] = max(W');
J1 = trace(A(Pi1,Pi1)*B);

% Solve the minimum
%[w,~,~,~,~] = linprog(cv,-eye(n*n),zeros(n*n,1),L,ones(2*n,1));
problem.f = cv;
[w,~,~,~,~] = linprog(problem);
% reshape w
W = reshape(w,n,n);
% Turn it into a 0/1 matrix
[~,Pi2] = max(W');
J2 = trace(A(Pi2,Pi2)*B);


% Choose the larger of J1 and J2

if (J1 >= J2)
    J = J1;
    Pi = Pi1;
    maxproblem = 1;
else
    J = J2;
    Pi = Pi2;
    maxproblem = 0;
end

end
