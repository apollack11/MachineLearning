function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    h = X*theta;
    errors = h - y;
    for i = 1:size(X,2)
        theta(i,1) = theta(i,1) - alpha*(1/m)*transpose(X(:,i))*errors;
    end
%     theta(1,1) = theta(1,1) - alpha*(1/m)*transpose(X(:,1))*errors;
%     theta(2,1) = theta(2,1) - alpha*(1/m)*transpose(X(:,2))*errors;


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
