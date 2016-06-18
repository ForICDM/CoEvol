function [ rmse ] = co_evolve_l1(data1, data2, data3, T, options)

two_nets = 0;
if isempty(data3)
    two_nets = 1;
end

%% parameter setting
addpath framework/
k = options.k;                 % latent dimensions
alfa = options.alfa;             % weight decay
beda = options.beda;             % weight decay for W0, W1, W2...
gama = options.gama;
fida = options.fida;             % time decay parameter
n = options.n;
theta = initialize(n, k);   % Randomly initialize the parameters
if two_nets
    theta = theta(1:5*n*k);
end
%fprintf('[%s] finish loading data... \n',datestr(now, 'mm/dd/yy HH:MM:SS'));
fprintf('[%s] matrix dimension: %d, param size: %d, testing |T|-1=%d\n', ...
    datestr(now, 'mm/dd/yy HH:MM:SS'), n, length(theta), T);

%% sample zero entries
zeros_len = data1(T).new_idx_avg_len;
rng(8)
x_idx = randi(n, zeros_len, 1);
rng(23)
y_idx = randi(n, zeros_len, 1);
zeros_idx = [x_idx, y_idx];

%% use minFunc to minimize the function
addpath optimize/
opt.Method = 'lbfgs';   % Here, we use L-BFGS to optimize our cost function. 
opt.maxIter = 100;      % Maximum number of iterations of L-BFGS to run 
opt.display = 'on';
opt.DerivativeCheck = 'off';
opt.TolFun  = 1e-8;
opt.TolX = 1e-9;

%filename = sprintf('./data/tmp/dblp_T%d_opttheta.mat', T);
filename = sprintf('./data/tmp/%s_T%d_k%d_fida%g_opttheta_l1.mat', options.datasets, T, k, fida);

if exist(filename,'file')
    opttheta = importdata(filename);
else
    [opttheta, ~] = minFunc( @(p) graph_cost_l1(p, k, alfa, beda, gama, T, n, fida, data1, data2, data3, zeros_idx), theta, opt);
    save(filename, 'opttheta');
end

%% result analysis
U = reshape(opttheta(1:n*k), n, k);
if two_nets
    X1 = reshape(opttheta(n*k+1:2*n*k), n, k);
    X2 = reshape(opttheta(2*n*k+1:3*n*k), n, k);
    Y1 = reshape(opttheta(3*n*k+1:4*n*k), n, k);
    Y2 = reshape(opttheta(4*n*k+1:5*n*k), n, k);
    V1 = X1 + (T+1) * Y1;
    V2 = X2 + (T+1) * Y2;
    V = V1 + V2;
    nonzeros_idx1 = data1(T+1).idx;
    nonzeros_idx2 = data2(T+1).idx;
    [all_idx, ~, ~] = union([nonzeros_idx1;nonzeros_idx2], zeros_idx, 'rows', 'stable');
else
    X1 = reshape(opttheta(n*k+1:2*n*k), n, k);
    X2 = reshape(opttheta(2*n*k+1:3*n*k), n, k);
    X3 = reshape(opttheta(3*n*k+1:4*n*k), n, k);
    Y1 = reshape(opttheta(4*n*k+1:5*n*k), n, k);
    Y2 = reshape(opttheta(5*n*k+1:6*n*k), n, k);
    Y3 = reshape(opttheta(6*n*k+1:7*n*k), n, k);
    V1 = X1 + (T+1) * Y1;
    V2 = X2 + (T+1) * Y2;
    V3 = X3 + (T+1) * Y3;
    V = V1 + V2 + V3;
    nonzeros_idx1 = data1(T+1).idx;
    nonzeros_idx2 = data2(T+1).idx;
    nonzeros_idx3 = data3(T+1).idx;
    [all_idx, ~, ~] = union([nonzeros_idx1;nonzeros_idx2;nonzeros_idx3], zeros_idx, 'rows', 'stable');
end
sparse_result = zeros(length(all_idx), 3);

for i = 1:length(all_idx)
    idx_i = all_idx(i, 1);
    idx_j = all_idx(i, 2);
    sparse_result(i,:) = [idx_i, idx_j, U(idx_i,:) * V(idx_j, :)'];
end

outputs = sparse(sparse_result(:,1), sparse_result(:,2), sparse_result(:,3), n, n);
if two_nets
    errors = (data1(T+1).mat + data2(T+1).mat) - outputs;
else
    errors = (data1(T+1).mat + data2(T+1).mat + data3(T+1).mat) - outputs;
end
rmse = sqrt(sum(sum(errors .* errors))/length(sparse_result));
rmse = full(rmse);
