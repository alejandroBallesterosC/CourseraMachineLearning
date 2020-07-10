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
%NN: Input Layer --> Hidden Layer --> Output Layer

%Feed Forward
A1 = [ones(m,1) X];%activations of input layer are the inputs| [5,000x401]
Z2 = A1*transpose(Theta1);%compute Z2 to compute activations of second layer| [5000x401]x[401x25] = [5000x25]
A2 = [ones(m,1) sigmoid(Z2)];%compute activations of second layer, add bias units| [5000x26]
Z3 = A2*transpose(Theta2);%compute Z3 to compute activations of output layer AKA outputs| [5000x26]x[26x10]=[5000x10]
A3 = sigmoid(Z3);%Outputs AKA activations of output layer| [5000x10]
H = A3;
%Create a y_cost matrix that maps the given y vector containing values between
%1-10 to a matrix of row vectors containing zeros and ones
y_cost = [zeros(1, y(1)-1) 1 zeros(1, num_labels-y(1))];
for i= 2:length(y)
    y_temp = [zeros(1, y(i)-1) 1 zeros(1, num_labels-y(i))];
    y_cost = [y_cost; y_temp];
end

%Calculate Cost Function (J)
for k = 1:num_labels
    for i = 1:m
    J = J+(-y_cost(i,k)*log(H(i,k))-(1-y_cost(i,k))*log(1-H(i,k)));
    end
end
J = (1/m)*J;%unregularized

%so you don't regularize using the first column of thetas 
%(thetas that map from bias unit in current layer to all other units in next layer)
Theta1_reg = Theta1;
Theta2_reg = Theta2;
Theta1_reg(:,1) = 0;
Theta2_reg(:,1) = 0;

reg_term = sum(sum(Theta1_reg.^2))+sum(sum(Theta2_reg.^2));
reg_term = (lambda/(2*m))*reg_term;
J = J+reg_term;

%Calculate Gradients by using backpropagation algorithm
d3 = A3-y_cost;%[5000x10]
d2 = (d3*Theta2(:,2:end)).*sigmoidGradient(Z2);%[5000x25]
Delta1 = transpose(d2)*A1;%[25x401]
Delta2 = transpose(d3)*A2;%[10x26]
Theta1_grad = (1/m).*Delta1+(lambda/m).*Theta1_reg;%[25x401]
Theta2_grad = (1/m).*Delta2+(lambda/m).*Theta2_reg;%[10x26]

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
