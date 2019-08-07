% Comparison of QAP Algorithms between themselves -- no ground truth
% available

% Local File Directories:
%datadir = '../TestBed/data/gaussian/';
%datadir = '../TestBed/data/uniform/';
currdir = pwd;
%resdir = '../results/gaussian/';
%resdir = '../results/largen_uniform/';
resdir = '../results/largen_gaussian/';

% Methods:
%optmethod = 'ABMethod'; % Solves Linear Assignment Problem on C=AB : max trace(Pi*C)
%optmethod = 'GCN2b0'; % Uses Graph Convolutive Network with 2 layers, no bias
%optmethod = 'GCN2b1'; % Uses Graph Convolutive Network with 2 layers, with bias
%optmethod = 'GCN3b0'; % Uses the Graph Convolutive Network with 3 layers, no bias
%optmethod = 'GCN3b1'; % Uses the Graph Convolutive Network with 3 layers, with bias
%optmethod = 'Iterative'; % Iterates between LAPs

algos = {'ABMethod','Iterative','GCN2b0','GCN2b1','GCN3b0','GCN3b1'};
nvect = [100,150,200];
rankvect = zeros(1,1);
Nalg = length(algos);
Nrealiz = 100;
Nrank = length(rankvect);
Jobj = zeros(Nrank,Nrealiz,Nalg);

for in=1:length(nvect)
    
    n = nvect(in);
    rankvect(1) = n;

for ir = 1:Nrealiz
for irank = 1:Nrank
    r = rankvect(irank);

% Problem instance
U = randn(n,r);
A = U*U';

U = rand(n,r);
B = U*U';


for ia = 1:Nalg

    optmethod = algos{ia};
    
    %Parameters of various Methods
    % GCN2, GCN3, or Iterative
    switch optmethod
        case 'GCN2b0'
            % GCN with 2 layers and no bias
            L = 2;
            % Weights
            W1 = csvread('2_layers_noBias/w1.csv');
            W2 = csvread('2_layers_noBias/w2.csv');
            d = size(W1,1);
            n1 = size(W1,2);
            D = size(W2,2);
            Weight{1} = W1;
            Weight{2} = W2;
            % Biases
            b1 = zeros(1,n1);
            Bias{1} = b1;
            b2 = zeros(1,D);
            Bias{2}=b2;

        case 'GCN2b1'
            % GCN with 2 layers and with bias
            L = 2;
            % Weights
            W1 = csvread('2_layers_withBias/w1.csv');
            W2 = csvread('2_layers_withBias/w2.csv');
            d = size(W1,1);
            Weight{1} = W1;
            Weight{2} = W2;
            % Biases
            b1 = csvread('2_layers_withBias/b1.csv');
            Bias{1} = b1;
            b2 = csvread('2_layers_withBias/b2.csv');
            Bias{2} = b2;

        case 'GCN3b0'
            % GCN with 3 layers and no bias
            L = 3; % Number of Layers
            % Weights
            W1 = csvread('3_layers_noBias/w1.csv');
            W2 = csvread('3_layers_noBias/w2.csv');
            W3 = csvread('3_layers_noBias/w3.csv');
            d = size(W1,1);
            n1 = size(W1,2);
            n2 = size(W2,2);
            D = size(W3,2);
            Weight{1} = W1;
            Weight{2} = W2;
            Weight{3} = W3;
            % Biases
            b1 = zeros(1,n1);
            Bias{1} = b1;
            b2 = zeros(1,n2);
            Bias{2} = b2;
            b3 = zeros(1,D);
            Bias{3} = b3;

        case 'GCN3b1'
            % GCN with 3 layers and with bias
            L = 3; % Number of Layers
            % Weights
            W1 = csvread('3_layers_withBias/w1.csv');
            W2 = csvread('3_layers_withBias/w2.csv');
            W3 = csvread('3_layers_withBias/w3.csv');
            d = size(W1,1);
            Weight{1} = W1;
            Weight{2} = W2;
            Weight{3} = W3;
            % Biases
            b1 = csvread('3_layers_withBias/b1.csv');
            b2 = csvread('3_layers_withBias/b2.csv');
            b3 = csvread('3_layers_withBias/b2.csv');
            Bias{1} = b1;
            Bias{2} = b2;
            Bias{3} = b3;

        case 'Iterative'
            % Iterative Algorithm
            Niter = 10; % Maximum number of iterations
            epstol = 1e-6; % Tolerance for stopping criterion

        otherwise
            % Ignore
    end

    % Run on a method
    switch optmethod
        case 'ABMethod'
            [Jhat,Pihat,maxproblem] = ABNetwork(A,B,A,B);

        case {'GCN2b0','GCN2b1','GCN3b0','GCN3b1'}
            [Jhat,Pihat,maxproblem] = QAPGCN(A,B,Weight,Bias,L);

        case 'Iterative'
            [Jhat,Pihat,maxproblem] = QAPIterative(A,B,Niter,epstol);

    end


    Jobj(irank,ir,ia) = Jhat;
    
end  % for ia=1:Nalg
end % for irank=1:Nrank
end % for ir=1:Nrealiz

% Process results
TopAlg = zeros(Nalg,Nrank);
AvgBelow = zeros(Nalg,Nrank);

%
%TA = zeros(Nalg,1);
%Perf = zeros(Nalg,1);

for irank=1:Nrank
    % Order each realization and record the order
    for ir=1:Nrealiz
        v = reshape(Jobj(irank,ir,1:Nalg),Nalg,1);
        [vsort,isort] = sort(v,'descend');
        TopAlg(isort(1),irank) = TopAlg(isort(1),irank) + 1;
        AvgBelow(1:Nalg,irank) = AvgBelow(1:Nalg,irank) + (v-vsort(1))/vsort(1);
    end
end
TopAlg = TopAlg/Nrealiz;
AvgBelow = AvgBelow/Nrealiz;

TA = mean(TopAlg,2);
Perf = mean(AvgBelow,2);

figure(1)
calg = categorical(algos);
bar(calg,100*TA)
ylabel('%% Top Alg')
cd(resdir);
print(sprintf('TopAlg_n%d.jpg',n),'-djpeg100');
cd(currdir);

figure(2)
bar(calg,100*Perf)
ylabel('Perf from Max [%%]')
cd(resdir);
print(sprintf('AlgPerf_n%d.jpg',n),'-djpeg100');
cd(currdir);


for irank = 1:Nrank
    r = rankvect(irank);
    figure(1)
    calg = categorical(algos);
    bar(calg,100*TopAlg(:,irank))
    ylabel('%% Top Alg'),
    title(sprintf('Frequency of Top Algorithm, for data of rank %d',r)),
    
    cd(resdir);
    print(sprintf('TopAlg_n%dr%d.jpg',n,r),'-djpeg100');
    cd(currdir);

    figure(2)
    bar(calg,100*AvgBelow(:,irank))
    ylabel('Perf from Max [%%]'),
    title(sprintf('Algorithm Performance wrt the best solution, for data of rank %d',r)),
    cd(resdir);
    print(sprintf('AlgPerf_n%dr%d.jpg',n,r),'-djpeg100');
    cd(currdir);
end

end  % for in=1:length(nvect)
