% WARNING: THIS CODE NEEDS THE DEEP LEARNING TOOLBOX
clear
close all
rng(0);

% Settings: Create some data
N_test = 10000;
Ntrain = 10000;
dim = 3; % Dimension for this example
C1 = 1;
C2 = 1;
mu1 = 0;
mu2 = 0.5;
sigma1 = 1;
sigma2 = 1;

% Loss function: Mean Squared Error
loss_mse = @(y_true, y_pred) mean((y_true - y_pred).^2);

% Arrays to store results
array_err_train = zeros(1, 1);
array_err_test = zeros(1, 1);
array_time_train = zeros(1, 1);
array_time_predict = zeros(1, 1);
ratio = 0.9;

disp(dim);

% Define function
f_func = @(x) (C1 * exp(-vecnorm(x - mu1, 2, 2).^2 / sigma1) + ...
               C2 * exp(-vecnorm(x - mu2, 2, 2).^2 / sigma2)) .* ...
              vecnorm(x, 2, 2).^2;

% Get training and test set: [-1, 1]^d
sobol_seq = sobolset(dim, 'Skip', 1e3, 'Leap', 1e2); % Create Sobol set with specified parameters
sample = 2 * sobol_seq(1:Ntrain, :) - 1; % Generate sample points and scale to [-1, 1]
shuffle = randperm(Ntrain); % Randomly shuffle indices
sample = sample(shuffle, :); % Shuffle the samples

X_train = sample(1:Ntrain, :); % Training data
X_train(1, :) = mu1; % Set specific rows to mu1
X_train(2, :) = mu2; % Set specific rows to mu2

X_test = 2 * rand(N_test, dim) - 1; % Test data

y_train = f_func(X_train); % Compute training outputs
y_test = f_func(X_test); % Compute test outputs

% Define the neural network architecture
layers = [
    featureInputLayer(dim, 'Normalization', 'none', 'Name', 'input')
    fullyConnectedLayer(512, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(512, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(512, 'Name', 'fc3')
    reluLayer('Name', 'relu3')
    fullyConnectedLayer(512, 'Name', 'fc4')
    reluLayer('Name', 'relu4')
    fullyConnectedLayer(1, 'Name', 'output')
    regressionLayer('Name', 'regressionoutput')
    ];

% Set training options
options = trainingOptions('adam', ...
    'MaxEpochs', 100, ...
    'InitialLearnRate', 0.005, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 20, ...
    'LearnRateDropFactor', 0.9, ...
    'GradientThreshold', 1, ...
    'ValidationData', {X_train(ratio * Ntrain + 1:end, :), y_train(ratio * Ntrain + 1:end)}, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', false, ...
    'ValidationPatience', 10);

% Train the neural network
tic;
net = trainNetwork(X_train(1:ratio * Ntrain, :), y_train(1:ratio * Ntrain), layers, options);
t_train = toc;

% Compute predictions
tic;
y_train_pred = predict(net, X_train);
y_test_pred = predict(net, X_test);
t_pred = toc;

% Compute errors
err_train = loss_mse(y_train, y_train_pred);
err_test = loss_mse(y_test, y_test_pred);

% Store results
array_err_train(1) = err_train;
array_err_test(1) = err_test;
array_time_train(1) = t_train;
array_time_predict(1) = t_pred;

% Display results
fprintf('Training Error (MSE): %.4e\n', err_train);
fprintf('Test Error (MSE): %.4e\n', err_test);
fprintf('Training Time: %.2f seconds\n', t_train);
fprintf('Prediction Time: %.2f seconds\n', t_pred);
