function theta = initialize(n, k)

    %% Initialize parameters randomly based on layer sizes.

    % we'll choose weights uniformly from the interval [0, r]
    r  = sqrt(6) / sqrt(n+k+1); 
    %r  = 1;
    U  = rand(n, k) * r;
    X1 = rand(n, k) * r;
    X2 = rand(n, k) * r;
    X3 = rand(n, k) * r;
    Y1 = rand(n, k) * r;
    Y2 = rand(n, k) * r;
    Y3 = rand(n, k) * r;

    % Convert weights and bias gradients to the vector form.
    % This step will "unroll" (flatten and concatenate together) all 
    % your parameters into a vector, which can then be used with minFunc. 
    theta = [U(:); X1(:); X2(:); X3(:); Y1(:); Y2(:); Y3(:)];
end

