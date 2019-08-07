function [Jhat,Pihat,maxproblem] = QAPGCN(A,B,Weight,Bias,L)
% QAPGCN Quadratic Assignment Problem using a Graph Convolutional Network
% Use:
%   [Jhat,Pihat,maxproblem] = QAPGCN(A,B,Weight,Bias,L)
%
% Radu Balan, 17 July 2019

n = size(A,1);
d = size(Weight{1},1);

% Format Data
X1 = A/norm(A,'fro'); % Data normalized by Frobenious norm
X2 = B/norm(B,'fro'); % Date normalized by the Frobenious norm

X = [X1;X2];% Do you need to adjust the columns of X to fit size(W{1},1) ?
if size(X,2) < d
    % Insert 0 columns
    X = [X,zeros(2*n,d-n)];
elseif size(X,2) > d
    % Discard columns
    X = X(:,1:d);
end

Adj = eye(2*n) + [zeros(n),X1*X2;X2*X1,zeros(n)];
for k=1:L
    b{k} = ones(2*n,1)*Bias{k};
end
%L : Number of GCN layers
%W : L-array of dk x d(k+1) weights
%b : L-array of 2n x dk biases
% GCN Filter
Y = GCNRep(X,Adj,Weight,b,L);
% Call the LAP
Y1 = Y(1:n,:);
Y2 = Y(1+n:2*n,:);
[Jhat,Pihat,maxproblem] = ABNetwork(Y1,Y2,A,B);
end

