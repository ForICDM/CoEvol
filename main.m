clear;

%% test dm and db
dataset = datasets{1};
dm = importdata('data/dblp_dmdb_dm.mat');
data1 = dm.dm_net;
n = dm.n;
db = importdata('data/dblp_dmdb_db.mat');
data2 = db.db_net;
data3 = [];

rmse_dmdb = zeros(length(data1) - 1, 1);
    
% parameter setting
options.datasets = dataset;
options.n = n;
options.alfa = 0.1;
options.beda = 0.1;
options.gama = 0.1;
options.k = 10;
options.fida = 0.3;

for T = 1:(length(data1) - 1)
    fprintf('[%s] |T| = %g\n', datestr(now, 'mm/dd/yy HH:MM:SS'), T);
    rmse = co_evolve(data1, data2, data3, T, options);
    fprintf('\n[%s] RMSE=%.4f\n', datestr(now, 'mm/dd/yy HH:MM:SS'), rmse);
    rmse_dmdb(T) = rmse;
end
save('./data/plots/dmdb_our.mat','rmse_dmdb');

%% test dm and ml
dataset = datasets{2};
dm = importdata('data/dblp_dmml_dm.mat');
data1 = dm.dm_net;
n = dm.n;
ml = importdata('data/dblp_dmml_ml.mat');
data2 = ml.ml_net;
data3 = [];

rmse_dmml = zeros(length(data1) - 1, 1);
    
% parameter setting
options.datasets = dataset;
options.n = n;
options.alfa = 0.1;
options.beda = 0.1;
options.gama = 0.1;
options.k = 10;
options.fida = 0.3;

for T = 1:(length(data1) - 1)
    fprintf('[%s] |T| = %g\n', datestr(now, 'mm/dd/yy HH:MM:SS'), T);
    rmse = co_evolve(data1, data2, data3, T, options);
    fprintf('\n[%s] RMSE=%.4f\n', datestr(now, 'mm/dd/yy HH:MM:SS'), rmse);
    rmse_dmml(T) = rmse;
end
save('./data/plots/dmdb_our.mat','rmse_dmml');

%% test dm, ml and db
dataset = datasets{3};
dm = importdata('data/dblp_dmmldb_dm.mat');
data1 = dm.dm_net;
n = dm.n;
db = importdata('data/dblp_dmmldb_db.mat');
data2 = db.db_net;
ml = importdata('data/dblp_dmmldb_ml.mat');
data3 = ml.ml_net;

rmse_dmmldb = zeros(length(data1) - 1, 1);
    
% parameter setting
options.datasets = dataset;
options.n = n;
options.alfa = 0.1;
options.beda = 0.1;
options.gama = 0.1;
options.k = 10;
options.fida = 0.3;

for T = 1:(length(data1) - 1)
    fprintf('[%s] |T| = %g\n', datestr(now, 'mm/dd/yy HH:MM:SS'), T);
    rmse = co_evolve(data1, data2, data3, T, options);
    fprintf('\n[%s] RMSE=%.4f\n', datestr(now, 'mm/dd/yy HH:MM:SS'), rmse);
    rmse_dmmldb(T) = rmse;
end
save('./data/plots/dmdb_our.mat','rmse_dmmldb');
    
