






% THIS IS A FINAL FILE FOR THE KERNEL COMPUTATIONS WITHIN 4.3


rng(0);

% Settings      Create some data - do not use the matlab arrays
N_test = 10000;
N_train = 10000;

C1 = 1;
C2 = 1;
mu1 = 0;
mu2 = .5;
sigma1 = 1;
sigma2 = 1;



loss_rel = @(y_true, y_pred) sqrt(norm(y_true - y_pred).^2 / norm(y_true).^2);



% arrays to store results
array_err_train = zeros(16, 1);
array_err_test = zeros(16, 1);
array_time_train = zeros(16, 1);
array_time_predict = zeros(16, 1);


for dim=3

	disp(dim);


    name_method = 'Mat2, 1/2, 10k';			% this information is not used
	shape_para = 1/2 * 1/ sqrt(dim);




	f_func = @(x) (C1 * exp(-vecnorm(x - mu1, 2, 2).^2 / sigma1) + C2 * exp(-vecnorm(x - mu2, 2, 2).^2 / sigma2)) .* vecnorm(x, 2, 2).^2;
    

    % Get training and test set: [-1, 1]^d
	p = sobolset(dim);

	sample = 2*net(p, N_train) - 1;
	shuffle = randperm(N_train);
	sample = sample(shuffle, :);

    % X_train = np.random.rand(N_train, dim)
    X_train = sample(shuffle(1:N_train), :);
    X_train(1, :) = mu1;
    X_train(2, :) = mu2;

    X_test = 2*rand(N_test, size(X_train, 2)) - 1;

    y_train = f_func(X_train);
    y_test = f_func(X_test);

    
    % TODO: I CHANGED THE FOLLOWING LINES
    % Set a NN
    numInputs = dim;
    numLayers = 5;

    net_nn = network(numInputs,numLayers)

    net_nn.layers{1}.transferFcn = 'poslin';
    net_nn.layers{2}.transferFcn = 'poslin';
    net_nn.layers{3}.transferFcn = 'poslin';
    net_nn.layers{4}.transferFcn = 'poslin';

    net_nn = init(net_nn);
    
    % I DO NOT UNDERSTAND HOW TO SET THE WIDTHS OF THE LAYERS? 
    % THE ONLY POSSIBILITY I FOUND WAS TO USE feedforwardnet([512, 512, 512, 512], ),
    % HOWEVER HERE I DO NOT KNOW HOW TO SET THE REMAINING THINGS

    % # TODO: No clue how to implement Early stopping and learning rate scheduling

    % Set training option - use last 20% of training data for validation
    options = trainingOptions("adam", MaxEpochs=100, InitialLearnRate=0.005, ...
        GradientThreshold=1, ...
        ValidationData={X_train(.8*N_train:end, :), y_train(.8*N_train:end, :)}, ...
        Shuffle = "every-epoch", ...
        Plots="training-progress", ...
        Metrics="mse", ...
        Verbose=false);

    % Train the NN
    tic;
    net_nn = trainnet(X_train(1:.8*Ntrain, :), y_train(1:.8*Ntrain, :), net_nn, "mse", options);
    t_train = toc;

    % Compute predicitons of NN
    y_train_pred = net_nn(X_train(1:.8*Ntrain, :));
    tic;
    y_test_pred = net_nn(X_test);
    t_pred = toc;

    


    res_train_1L = abs(y_train_pred(:) - y_train(:));
    res_test_1L = abs(y_test_pred(:) - y_test(:));

        
    % Compute errors
    max_train = max(res_train_1L);
    err_train = loss_rel(y_train, y_train_pred);
    err_test = loss_rel(y_test, y_test_pred);

    array_err_train(dim, 1) = err_train;
    array_err_test(dim, 1) = err_test;
    array_time_train(dim, 1) = t_train;
    array_time_predict(dim, 1) = t_pred;


end















