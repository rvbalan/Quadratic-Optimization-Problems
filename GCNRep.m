function Y = GCNRep(X,Adj,W,b,L)
% GCNRep Graph Convolutive Network based Representation
% Use:
%  Y = GCNRep(X,Adj,W,b,L)
% Inputs:
%  X is a Nxd data matrix, where N is the number of vertices
%  Adj is a NxN adjacency matrix
%  W is an array of L weight matricesL W{1} of size d x n1, W{2} of size 
%    n1 x n2, ... W{L} of size n(L-1) x D
%  b is a L-array of biases b{1} matrix of size N x n1,..., b{L} matrix of
%    size NxD
%  L : number of layers
% Outputs:
%  Y is a NxD matrix of latent features
%
% 14 July 2019, Radu Balan

Y = X; % Or tailor to Nxd size where d=size(W{1},1)
for k=1:L-1
    Weight = W{k};
    bias = b{k};
    % Process
    Y = relufun(Adj*Y*Weight+bias);
end

Weight = W{L};
bias = b{L};
% Process
Y = Adj*Y*Weight+bias;

end
