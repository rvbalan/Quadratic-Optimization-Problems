% This driver/script solves the Quadratic Optimization Problem max tr(Pi*A*Pi'*B) using a specific
% method, and then processes and reports statistics of the results

% Local File Directories:
%datadir = '../TestBed/data/gaussian/';
datadir = '../TestBed/data/uniform/';
currdir = pwd;
%resdir = '../results/gaussian/';
resdir = '../results/uniform/';

% Methods:
%optmethod = 'ABMethod'; % Solves Linear Assignment Problem on C=AB : max trace(Pi*C)
%optmethod = 'GCN2b0'; % Uses Graph Convolutive Network with 2 layers, no bias
%optmethod = 'GCN2b1'; % Uses Graph Convolutive Network with 2 layers, with bias
%optmethod = 'GCN3b0'; % Uses the Graph Convolutive Network with 3 layers, no bias
optmethod = 'GCN3b1'; % Uses the Graph Convolutive Network with 3 layers, with bias
%optmethod = 'Iterative'; % Iterates between LAPs

% General Parameters of precomputed QAP problems
nvect = (2:10);
Nexm = 1000;
qmax = 50; % Maximum number of solutions

%Parameters of various Methods
% GCN2 or GCN3
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

    otherwise
        % Ignore
end

% Iterative Method:
Niter = 10; % Maximum number of iterations
epstol = 1e-6; % Tolerance for stopping criterion

DiffQAP = zeros(nvect(end),1);
RelDiffQAP = zeros(nvect(end),1);
ProbCorrect = zeros(nvect(end),1);
RankSol = zeros(nvect(end),1);
LAPMax = zeros(nvect(end),1);

DJQAP = zeros(nvect(end));
RDJQAP = zeros(nvect(end));
OptPiFound = zeros(nvect(end));
MethodRank = zeros(nvect(end));
FullLAPMax = zeros(nvect(end));


fprintf(1,'n: ');
for n = nvect(1):nvect(end)
    fprintf(1,' %d',n);
    datandr = sprintf('%sn%d',datadir,n);
    rvect = (1:n);
    
    for r=rvect(1):rvect(end)

        DeltaJQAP = zeros(Nexm,1);
        RelDeltaJQAP = zeros(Nexm,1);
        QualityMethod = zeros(Nexm,1);
        FindPi = zeros(Nexm,1);
        MaxProblem = zeros(Nexm,1);

        for ie = 1:Nexm
            
%            A = zeros(n);
%            B = zeros(n);
            
            % Load data
            cd(datandr);
            [fid,message1] = fopen(sprintf('A%dr%d.txt',ie,r),'r');
            Av = fscanf(fid,'%f');
            A = reshape(Av,n,n);
            fclose(fid);

            [fid,message2] = fopen(sprintf('B%dr%d.txt',ie,r),'r');
            Bv = fscanf(fid,'%f');
            B = reshape(Bv,n,n);
            fclose(fid);
            
            % Load precomputed results
            [fid,message3] = fopen(sprintf('P%dr%d.txt',ie,r),'r');
            Piv = fscanf(fid,'%d');
            ntop = length(Piv)/n;
            OptimPi = (reshape(Piv,n,ntop))';
            fclose(fid);

            [fid,message4] = fopen(sprintf('Obj%dr%d.txt',ie,r),'r');
            OptimObj = fscanf(fid,'%f');
            fclose(fid);
            cd(currdir);
            
            % Run on a method
            switch optmethod
                case 'ABMethod'
                    [Jhat,Pihat,maxproblem] = ABNetwork(A,B,A,B);
            
                case {'GCN2b0','GCN2b1','GCN3b0','GCN3b1'}
                    [Jhat,Pihat,maxproblem] = QAPGCN(A,B,Weight,Bias,L);

                case 'Iterative'
                    [Jhat,Pihat,maxproblem] = QAPIterative(A,B,Niter,epstol);

            end

            % Process results
            DeltaJQAP(ie) = OptimObj(1) - Jhat;
            RelDeltaJQAP(ie) = (OptimObj(1) - Jhat)/OptimObj(1);
            FindPi(ie) = (sum(abs(Pihat - OptimPi(1,:))) == 0);
            QualityMethod(ie) = find((Jhat-1e-8) <= OptimObj,1,'last'); % Rank in first ntop solutions
            MaxProblem(ie) = maxproblem;
        end
        
        % Summarize statistics of the method (r,n)
        DJQAP(r,n) = mean(DeltaJQAP);
        RDJQAP(r,n) = mean(RelDeltaJQAP);
        OptPiFound(r,n) = mean(FindPi);
        MethodRank(r,n) = mean(QualityMethod);
        FullLAPMax(r,n) = mean(MaxProblem);
        
    end
    % Average across ranks
    DiffQAP(n) = mean(DJQAP(1:n,n));
    RelDiffQAP(n) = mean(RDJQAP(1:n,n));
    ProbCorrect(n) = mean(OptPiFound(1:n,n));
    RankSol(n) = mean(MethodRank(1:n,n));
    LAPMax(n) = mean(FullLAPMax(1:n,n));
    
end
fprintf(1,'\n');

% Display Results
figure(1)
plot(nvect,DiffQAP(nvect))
xlabel('n'),ylabel('E[Jopt-Jhat]')
filname = sprintf('DiffCrit_%s.jpg',optmethod);
cd(resdir);
print(filname,'-djpeg100');
cd(currdir);

figure(2)
plot(nvect,RelDiffQAP(nvect))
xlabel('n'),ylabel('E[(Jopt-Jhat)/Jopt]')
filname = sprintf('RelDiffCrit_%s.jpg',optmethod);
cd(resdir);
print(filname,'-djpeg100');
cd(currdir);

figure(3)
plot(nvect,ProbCorrect(nvect))
xlabel('n'),ylabel('Prob[Jopt=Jhat]')
filname = sprintf('ProbCorrect_%s.jpg',optmethod);
cd(resdir);
print(filname,'-djpeg100');
cd(currdir);

figure(4)
plot(nvect,RankSol(nvect))
xlabel('n'),ylabel('Rank[Jhat]')
filname = sprintf('RankSolution_%s.jpg',optmethod);
cd(resdir);
print(filname,'-djpeg100');
cd(currdir);

figure(5)
plot(nvect,LAPMax(nvect))
xlabel('n'),ylabel('Fraction of Max LAP')
filname = sprintf('MaxLAP_%s.jpg',optmethod);
cd(resdir);
print(filname,'-djpeg100');
cd(currdir);


% Analyze results for each rank
for r=1:nvect(end)

    nmin = max(r,nvect(1));
    % Display Results
    figure(1)
    plot(nmin:nvect(end),DJQAP(r,nmin:nvect(end)))
    xlabel('n'),ylabel(sprintf('E[Jopt-Jhat] for rank %d',r))
    filname = sprintf('DiffCrit_%s_r%d.jpg',optmethod,r);
    cd(resdir);
    print(filname,'-djpeg100');
    cd(currdir);

    figure(2)
    plot(nmin:nvect(end),RDJQAP(r,nmin:nvect(end)))
    xlabel('n'),ylabel(sprintf('E[(Jopt-Jhat)/Jopt] for rank %d',r))
    filname = sprintf('RelDiffCrit_%s_r%d.jpg',optmethod,r);
    cd(resdir);
    print(filname,'-djpeg100');
    cd(currdir);

    figure(3)
    plot(nmin:nvect(end),OptPiFound(r,nmin:nvect(end)))
    xlabel('n'),ylabel(sprintf('Prob[Jopt=Jhat] for rank %d',r))
    filname = sprintf('ProbCorrect_%s_r%d.jpg',optmethod,r);
    cd(resdir);
    print(filname,'-djpeg100');
    cd(currdir);

    figure(4)
    plot(nmin:nvect(end),MethodRank(r,nmin:nvect(end)))
    xlabel('n'),ylabel(sprintf('Rank[Jhat] for rank %d',r))
    filname = sprintf('RankSolution_%s_r%d.jpg',optmethod,r);
    cd(resdir);
    print(filname,'-djpeg100');
    cd(currdir);

    figure(5)
    plot(nmin:nvect(end),FullLAPMax(r,nmin:nvect(end)))
    xlabel('n'),ylabel(sprintf('Fraction of Max LAP for rank %d',r))
    filname = sprintf('MaxLAP_%s_r%d.jpg',optmethod,r);
    cd(resdir);
    print(filname,'-djpeg100');
    cd(currdir);
    
end
