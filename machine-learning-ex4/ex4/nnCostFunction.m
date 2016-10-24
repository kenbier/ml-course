function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

K = num_labels;
Y = eye(K)(y, :);

a_1 = [ones(m, 1), X]
a_2 = sigmoid(a_1*Theta1');
z_2 = (Theta1*a_1')';
a_2 = [ones(size(a_2, 1), 1), a_2]
a_3 = sigmoid(a_2*Theta2');

cost = sum((-Y .* log(a_3)) - ((1 - Y) .* log(1 - a_3)), 2);
J = (1 / m) * sum(cost);

Theta1_nobias = Theta1(:, 2:end);
Theta2_nobias = Theta2(:, 2:end);

reg = lambda/(2*m)*(sum(sumsq(Theta1_nobias)) + sum(sumsq(Theta2_nobias)));

J = J + reg;

d_3 = a_3 - Y;
d_2 = ((d_3*Theta2_nobias).*sigmoidGradient(z_2));

Theta1_grad = 1/m * d_2' * a_1;
Theta2_grad = 1/m * d_3' * a_2;

reg_2 = [zeros(rows(Theta2_nobias), 1) Theta2_nobias];
reg_1 = [zeros(rows(Theta1_nobias), 1) Theta1_nobias];

Theta2_grad = Theta2_grad + (lambda/m)*reg_2;
Theta1_grad = Theta1_grad + (lambda/m)*reg_1;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
