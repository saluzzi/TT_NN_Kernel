






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


for dim=3:16

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


	% Compute kernel model and prediction
	tic;
	A0 = kernel_mat(X_train, X_train, shape_para) + 1e-8 * eye(N_train);
    coeff = A0 \ y_train;
	t_train = toc;

    % Compute errors
	tic;
    y_test_pred = kernel_mat(X_test, X_train, shape_para) * coeff;
	t_pred = toc;
    y_train_pred = A0 * coeff;


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



function array_kernel = kernel_mat(x1, x2, shape_para)
	dist_matrix = pdist2(x1, x2);

	array_kernel = exp(-shape_para * dist_matrix) .* (3 + 3 * shape_para * dist_matrix + 1 * (shape_para * dist_matrix).^2);

end













