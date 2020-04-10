function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

X_positives=X(y==1,:);
X_negatives=X(y==0,:);
plot(X_positives(:,1),X_positives(:,2),'k+');
plot(X_negatives(:,1),X_negatives(:,2),'ko');

% =========================================================================
hold off;

end
