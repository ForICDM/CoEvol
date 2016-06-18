methods = {'wcn', 'waa', 'hplp', 'nnmf', 'cp', 'our'};
dm = importdata('./data/dblp_random01_dm.mat');
data1 = dm.dm_net;
n = dm.n;
db = importdata('./data/dblp_random01_db.mat');
data2 = db.db_net;


k = 10;

%% loop with T (timeframe)
for T = 2:10
    % output network 1
    file_name = sprintf('./data/plots/jan19_random_T%d.dot', T);
    fid = fopen(file_name, 'wt');
    % dot file format
    dot_head = 'graph G { node[margin="0.06,0,025" width=0.2 height=0.15 style=filled, fontsize=6, fontname="Helvetica", colorscheme=greens3, color=1];';
    dot_tail = '}';
    
    %% to dot
    % edges
    HeatMap(full(data1(T).mat + data2(T).mat))
    
    edges1 = data1(T).idx;
    edges2 = data2(T).idx;

    fprintf(fid, dot_head);
    for i=1:length(edges2)
        fprintf(fid, '\n%d -- %d [color=blue];', edges2(i,1:2));
    end
    for i=1:length(edges1)
        fprintf(fid, '\n%d -- %d [color=tomato];', edges1(i,1:2));
    end

    
    nodes = unique(edges1(:,1:2));
    for j=1:length(nodes)
        fprintf(fid, '\n%d [label="",shape=ellipse, width=.075, height=.05, style=filled,color=olivedrab,fontcolor=white];', nodes(j));
    end
    
    fprintf(fid, dot_tail);
    fclose(fid);
    
    %% save as svg
    dot_path = file_name;
    svg_path = sprintf('%s.svg', dot_path);
    cmd = sprintf('/usr/local/bin/neato -Tsvg -Gsize=5,7 %s -o %s', dot_path, svg_path);
    system(cmd);
end
