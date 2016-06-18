clc;clear;

%% loading data
addpath data/ loading/
% conf = importdata('data/dblp_conf_net.mat');
% data1 = conf.conf_net;
% n = conf.n;
% journal = importdata('data/dblp_journal_net.mat');
% data2 = journal.journal_net;

sample_rate = 0.1;
[data1, n] = load_infectious(sample_rate);
data2 = data1;
%data3 = [];
data3 = data1;

two_nets = 0;
if isempty(data3)
    two_nets = 1;
end

%% parameter setting
addpath framework/
k = 10;                 % latent dimensions
alfa = 0.1;             % weight decay
beda = 0.1;             % weight decay for W0, W1, W2...
gama = 0.1;
T = 2;                  % |T|
fida = 0.3;             % time decay parameter
theta = initialize(n, k);   % Randomly initialize the parameters
if two_nets
    theta = theta(1:5*n*k);
end
fprintf('[%s] finish loading data... \n',datestr(now, 'mm/dd/yy HH:MM:SS'));
fprintf('[%s] matrix dimension: %d, param size: %d, testing |T|-1=%d\n', ...
    datestr(now, 'mm/dd/yy HH:MM:SS'), n, length(theta), T);

%% sample zero entries
zeros_len = data1(T).new_idx_avg_len;
rng(8)
x_idx = randi(n, zeros_len, 1);
rng(23)
y_idx = randi(n, zeros_len, 1);
zeros_idx = [x_idx, y_idx];

% %% gradient checking
% [~,grad] = graph_cost(theta, k, alfa, beda, gama, T, n, fida, data1, data2, data3, zeros_idx);
% numgrad = check_grad( @(x) graph_cost(x,  k, alfa, beda, gama, T, n, fida, data1, data2, data3, zeros_idx), theta);
% % Compare numerically computed gradients with the ones obtained from backpropagation
% diff = norm(numgrad-grad)/norm(numgrad+grad);
% % Should be small. Usually less than 1e-9.
% fprintf('[%s] gradient differences: %g\n', datestr(now, 'mm/dd/yy HH:MM:SS'), diff)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% gradient checking for L1
[~,grad] = graph_cost_l1(theta, k, alfa, beda, gama, T, n, fida, data1, data2, data3, zeros_idx);
numgrad = check_grad( @(x) graph_cost_l1(x,  k, alfa, beda, gama, T, n, fida, data1, data2, data3, zeros_idx), theta);
% Compare numerically computed gradients with the ones obtained from backpropagation
diff = norm(numgrad-grad)/norm(numgrad+grad);
% Should be small. Usually less than 1e-9.
fprintf('[%s] gradient differences: %g\n', datestr(now, 'mm/dd/yy HH:MM:SS'), diff)

