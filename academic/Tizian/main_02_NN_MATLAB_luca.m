% WARNING: THIS CODE NEEDS THE DEEP LEARNING TOOLBOX
clear
close all
rng(0);

% Settings: Create some data
N_test = 10000;
Ntrain = 10000;

C1 = 1;
C2 = 1;
mu1 = 0;
mu2 = 0.5;
sigma1 = 1;
sigma2 = 1;

loss_rel = @(y_true, y_pred) sqrt(norm(y_true - y_pred).^2 / norm(y_true).^2);

% Arrays to store results
array_err_train = zeros(16, 1);
array_err_test = zeros(16, 1);
array_time_train = zeros(16, 1);
array_time_predict = zeros(16, 1);

for dim = 3:16
    disp(dim);

    shape_para = 1 / 2 * 1 / sqrt(dim);

    f_func = @(x) (C1 * exp(-vecnorm(x - mu1, 2, 2).^2 / sigma1) + ...
                   C2 * exp(-vecnorm(x - mu2, 2, 2).^2 / sigma2)) .* ...
                  vecnorm(x, 2, 2).^2;

    % Get training and test set: [-1, 1]^d
    %load ../../data.mat
    p = sobolset(dim);
    sample = 2 * net(p, Ntrain) - 1;
    shuffle = randperm(Ntrain);
    sample = sample(shuffle, :);

    X_train = sample(shuffle(1:Ntrain), :);
    X_train(1, :) = mu1;
    X_train(2, :) = mu2;

    X_test = 2 * rand(N_test, size(X_train, 2)) - 1;

    y_train = f_func(X_train);
    y_test = f_func(X_test);

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

    % Set minibatch size
    miniBatchSize = 128;

    % Set training options
    options = trainingOptions('adam', ...
        'MaxEpochs', 100, ...
        'InitialLearnRate', 0.005, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropPeriod', 20, ...
        'LearnRateDropFactor', 0.9, ...
        'GradientThreshold', 1, ...
        'ValidationData', {X_train(0.9 * Ntrain:end, :), y_train(0.9 * Ntrain:end, :)}, ...
        'MiniBatchSize', miniBatchSize, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', false, ...
        'ValidationPatience', 10);
    %        'Plots', 'training-progress', ...

    % Train the neural network
    tic;
    net_nn = trainNetwork(X_train(1:0.9 * Ntrain, :), y_train(1:0.9 * Ntrain, :), layers, options);
    t_train = toc

    % Compute predictions
    y_train_pred = predict(net_nn, X_train);
    tic;
    y_test_pred = predict(net_nn, X_test);
    t_pred = toc;

    % Compute errors
    res_train_1L = abs(y_train_pred - y_train);
    res_test_1L = abs(y_test_pred - y_test);

    max_train = max(res_train_1L);
    err_train = loss_rel(y_train, y_train_pred);
    err_test = loss_rel(y_test, y_test_pred)

    array_err_train(dim, 1) = err_train;
    array_err_test(dim, 1) = err_test;
    array_time_train(dim, 1) = t_train;
    array_time_predict(dim, 1) = t_pred;
end
