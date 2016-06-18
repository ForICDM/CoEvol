function numgrad = check_grad(J, theta)
% numgrad = computeNumericalGradient(J, theta)
% theta: a vector of parameters
% J: a function that outputs a real-number. Calling y = J(theta) will return the
% function value at theta. 
  
% % Initialize numgrad with zeros
% numgrad = zeros(size(theta));
% 
% epsilon = 10^-4;
% vectors = eye(size(theta,1));
% for i = 1:size(theta)
%     numgrad(i) = (J(theta + epsilon .* vectors(:,i)) - J(theta - epsilon .* vectors(:,i)))/(2 * epsilon);
% end
% 
% %% ---------------------------------------------------------------

numgrad = zeros(size(theta));
perturb = zeros(size(theta));
e = 1e-4;
for p = 1:numel(theta)
    %fprintf('>> checking gradient: %d/%d\n', p, numel(theta));
    % Set perturbation vector
    perturb(p) = e;
    loss1 = J(theta - perturb);
    loss2 = J(theta + perturb);
    % Compute Numerical Gradient
    numgrad(p) = (loss2 - loss1) / (2*e);
    perturb(p) = 0;
end

end
