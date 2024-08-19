clear
close all
rng(0); % Set the random seed

% Settings
N_test = 10000; % Increase to int(1e5)
Ntrain = 10000;

C1 = 1; C2 = 1;
mu1 = 0; mu2 = 0.5;
sigma1 = 1; sigma2 = 1;

loss_rel = @(y_true, y_pred) sqrt(norm(y_true - y_pred)^2 / norm(y_true)^2);

array_dim = 3:16;
% array_dim = 3:5;

dic_results = struct();

for dim = [3] % array_dim:
    f_func = @(x) (C1 * exp(-vecnorm(x - mu1, 2, 2).^2 / sigma1) ...
        + C2 * exp(-vecnorm(x - mu2, 2, 2).^2 / sigma2)) .* vecnorm(x, 2, 2).^2;

    % Get training and test set: [-1, 1]^d
    % Generate Sobol sequence
    sobol = sobolset(dim, 'Skip', 1e3, 'Leap', 1e2); % Create Sobol set with specified parameters
    num_points = 2^nextpow2(Ntrain); % Determine the number of points needed
    sample = 2 * sobol(1:num_points, :) - 1; % Generate sample points and scale to [-1, 1]

    % Shuffle and select training samples
    shuffle = randperm(size(sample, 1)); % Randomly shuffle indices
    X_train = sample(shuffle(1:Ntrain), :); % Select N_train samples

    % Set specific rows of X_train to mu1 and mu2
    X_train(1, :) = mu1;
    X_train(2, :) = mu2;

    X_test = 2 * rand(N_test, dim) - 1;

    y_train = f_func(X_train);
    y_test = f_func(X_test);

    % Set up data for NN training
    N_data = size(X_train, 1);
    fraction_train = 0.9;

    % Convert to tables (for consistency, but using numeric arrays for trainNetwork)
    train_data = [X_train(1:round(fraction_train*N_data), :), y_train(1:round(fraction_train*N_data))];
    val_data = [X_train(round(fraction_train*N_data)+1:end, :), y_train(round(fraction_train*N_data)+1:end)];

    X_train_numeric = train_data(:, 1:dim);
    y_train_numeric = train_data(:, end);
    X_val_numeric = val_data(:, 1:dim);
    y_val_numeric = val_data(:, end);

    % Define the neural network layers
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

    % Define the training options
    options = trainingOptions('adam', ...
        'MaxEpochs', 100, ...
        'InitialLearnRate', 5e-3, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropPeriod', 20, ...
        'LearnRateDropFactor', 0.9, ...
        'ValidationData', {X_val_numeric, y_val_numeric}, ...
        'ValidationFrequency', 30, ...
        'ValidationPatience', 10, ...
        'Plots', 'training-progress');

    % Train the network
    t0_train = tic;
    net = trainNetwork(X_train_numeric, y_train_numeric, layers, options);
    t1_train = toc(t0_train);

    % Predict
    t0_predict = tic;
    y_train_pred = predict(net, X_train_numeric);
    y_test_pred = predict(net, X_test);
    t1_predict = toc(t0_predict);

    % Compute errors
    res_train_1L = abs(y_train_pred - y_train);
    res_test_1L = abs(y_test_pred - y_test);

    max_train = max(res_train_1L);
    err_train = loss_rel(y_train, y_train_pred);
    err_test = loss_rel(y_test, y_test_pred);

    dic_results.NN.err_train = [dic_results.NN.err_train; err_train];
    dic_results.NN.err_test = [dic_results.NN.err_test; err_test];
    dic_results.NN.time_train = [dic_results.NN.time_train; t1_train];
    dic_results.NN.time_predict = [dic_results.NN.time_predict; t1_predict];
end

% Plot results
figure;
plot(array_dim, dic_results.NN.err_test, 'x--');
title('err_test');
set(gca, 'YScale', 'log');
xlabel('dim');
ylabel('Error');
legend('NN');
