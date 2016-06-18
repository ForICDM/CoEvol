function [cost,grad] = graph_cost(theta, k, alfa, beda, gama, T, n, fida, data1, data2, data3, zeros_idx)
% theta is a one dimension vector with U{:}, X{:}, Y{:}, ...

two_nets = 0;
if isempty(data3)
    two_nets = 1;
end

if two_nets
    U = reshape(theta(1:n*k), n, k);
    X1 = reshape(theta(n*k+1:2*n*k), n, k);
    X2 = reshape(theta(2*n*k+1:3*n*k), n, k);
    Y1 = reshape(theta(3*n*k+1:4*n*k), n, k);
    Y2 = reshape(theta(4*n*k+1:5*n*k), n, k);
else
    U = reshape(theta(1:n*k), n, k);
    X1 = reshape(theta(n*k+1:2*n*k), n, k);
    X2 = reshape(theta(2*n*k+1:3*n*k), n, k);
    X3 = reshape(theta(3*n*k+1:4*n*k), n, k);
    Y1 = reshape(theta(4*n*k+1:5*n*k), n, k);
    Y2 = reshape(theta(5*n*k+1:6*n*k), n, k);
    Y3 = reshape(theta(6*n*k+1:7*n*k), n, k);
end

% gradient variables initialize them to zeros 
Ugrad = zeros(size(U)); 
X1grad = zeros(size(X1));
X2grad = zeros(size(X2));
Y1grad = zeros(size(Y1)); 
Y2grad = zeros(size(Y2)); 

if ~two_nets
    X3grad = zeros(size(X3));
    Y3grad = zeros(size(Y3)); 
end

% least squares component of cost with decay function D(t)
leastsquares = 0;
for t = 1:T
    % network one
    nonzeros_idx = data1(t).idx;
    all_idx = union(zeros_idx, nonzeros_idx, 'rows');
    sparse_result = zeros(length(all_idx), 3);
    for i = 1:length(all_idx)
        idx_i = all_idx(i, 1);
        idx_j = all_idx(i, 2);
        sparse_result(i,:) = [idx_i, idx_j, U(idx_i,:) * (X1(idx_j, :) + t * Y1(idx_j, :))'];
    end
    outputs = sparse(sparse_result(:,1), sparse_result(:,2), sparse_result(:,3), n, n);
    errors1 =  data1(t).mat - outputs; 
    
    % network two
    nonzeros_idx = data2(t).idx;
    all_idx = union(zeros_idx, nonzeros_idx, 'rows');
    sparse_result = zeros(length(all_idx), 3);
    for i = 1:length(all_idx)
        idx_i = all_idx(i, 1);
        idx_j = all_idx(i, 2);
        sparse_result(i,:) = [idx_i, idx_j, U(idx_i,:) * (X2(idx_j, :) + t * Y2(idx_j, :))'];
    end
    outputs = sparse(sparse_result(:,1), sparse_result(:,2), sparse_result(:,3), n, n);
    errors2 = data2(t).mat - outputs; 
    
    if ~two_nets
        nonzeros_idx = data3(t).idx;
        all_idx = union(zeros_idx, nonzeros_idx, 'rows');
        sparse_result = zeros(length(all_idx), 3);
        for i = 1:length(all_idx)
            idx_i = all_idx(i, 1);
            idx_j = all_idx(i, 2);
            sparse_result(i,:) = [idx_i, idx_j, U(idx_i,:) * (X3(idx_j, :) + t * Y3(idx_j, :))'];
        end
        outputs = sparse(sparse_result(:,1), sparse_result(:,2), sparse_result(:,3), n, n);
        errors3 = data3(t).mat - outputs;   
    end
    
    Dt = exp(-fida*(T-t));
    leastsquares = leastsquares + (Dt/2)*sum(sum(errors1 .* errors1)) + (Dt/2)*sum(sum(errors2 .* errors2));
    
    if ~two_nets
        leastsquares = leastsquares + (Dt/2)*sum(sum(errors3 .* errors3));
    end
    
    % derivatives
    delta1 = Dt * errors1;
    delta2 = Dt * errors2;
    if ~two_nets
        delta3 = Dt * errors3;
    end
    X1grad = X1grad + delta1' * (-U);
    X2grad = X2grad + delta2' * (-U);
    Y1grad = Y1grad + delta1' * (-t*U);
    Y2grad = Y2grad + delta2' * (-t*U);
    Ugrad = Ugrad + delta1 * (-X1 - t * Y1) + delta2 * (-X2 - t * Y2);
    if ~two_nets
        X3grad = X3grad + delta3' * (-U);
        Y3grad = Y3grad + delta3' * (-t*U);
        Ugrad = Ugrad + delta3 * (-X3 - t * Y3);
    end

end

weightdecay = alfa/2 * sum(sum(U .* U)) + beda/2 * sum( sum(X1 .* X1)) + gama/2 * sum( sum(Y1 .* Y1)) ...
     + beda/2 * sum( sum(X2 .* X2)) + gama/2 * sum( sum(Y2 .* Y2));

if ~two_nets
    weightdecay = weightdecay + beda/2 * sum( sum(X3 .* X3)) + gama/2 * sum( sum(Y3 .* Y3));
end

% gradient
Ugrad = Ugrad + alfa * U;
X1grad = X1grad + beda * X1;
X2grad = X2grad + beda * X2;
Y1grad = Y1grad + gama * Y1;
Y2grad = Y2grad + gama * Y2;

% gradient vectorization
grad = [Ugrad(:); X1grad(:); X2grad(:); Y1grad(:); Y2grad(:)];
cost = leastsquares + weightdecay;

if ~two_nets
    X3grad = X3grad + beda * X3;
    Y3grad = Y3grad + gama * Y3;
    grad = [Ugrad(:); X1grad(:); X2grad(:); X3grad(:); Y1grad(:); Y2grad(:); Y3grad(:)];  
end

end