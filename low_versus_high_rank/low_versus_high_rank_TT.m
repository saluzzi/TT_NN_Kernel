%clear
close all

addpath ..\Functions\TT
check_tt

for d = 16
isLowRank = 0;
if isLowRank
    fun = @(x) exp(-sum(x)/(2*d));
    dfun = @(x,i) -exp(-sum(x)/(2*d))/(2*d);
else
    fun = @(x) exp(-prod(x));
    dfun = @(x,i) -prod(x(1:i-1))*prod(x(i+1:end))*exp(-prod(x));
end

% fun = @(x) exp(-norm(x(2:end))+x(1)-1);
%fun = @(x) x*P0*x';
% mu = [0 1];
% sigma = [1 1];
% fun = @(x) exp(-sum((x-mu(1)).^2)/sigma(1))+exp(-sum((x-mu(2)).^2)/sigma(2));
tol = 1.e-5;


a = 0;
b = 1;
%%

tic
lambda = 0;
n = 7;
[x,~] = lgwt(n,a,b);
% Basis functions and theri derivatives
[P,dP] = legendre_rec(x,a,b,numel(x)-1);
% Cell array of vectors of grid points
repx = repmat({x},d,1);
% Cell array of basis functions and their derivatives
repP = repmat({cat(d,P,dP)},d,1);
% TT matrix of the basis functions
mkronP = mtkron([{tt_matrix(P)} repmat({tt_matrix(P)},1,d-1)]) ;
[V_sl,eval,Sample] = gradient_cross_new_sample(repx, repP,@(x) fun2(x,fun,dfun), tol, lambda,0,'y0',1,'kickrank',1);
V = core2cell(V_sl);
CPU(d-2) = toc;
Sample2 = unique(Sample,'rows');
N_samples = size(Sample2,1);
dof = 0;
for i =1:d
    dof = dof+numel(V{i});
end

%%

Value_sampled = fun2(Sample2,fun,dfun);
[Vx] = value_function(Sample2,a,b,V);
err_train_inf = max(abs(Vx-Value_sampled(:,1)));
err_train_2 = norm(Vx-Value_sampled(:,1))/norm(Value_sampled(:,1))
norm(Vx-Value_sampled(:,1))/N_samples

%%
NN = 1.e5;
X_test = (b-a)*rand(NN,d)+a;
Value_test = fun2(X_test,fun,dfun);
[Vx_test] = value_function(X_test,a,b,V);
err_test_inf = max(abs(Vx_test-Value_test(:,1)));
err_test_2(d-2) = norm(Vx_test-Value_test(:,1))/norm(Value_test(:,1))
mse(d-2) = sum((Vx_test-Value_test(:,1)).^2)/NN;

%[num2str(err_train_inf) ' & ' sprintf('%2e',err_test_inf) ' & ' num2str(err_train_2) ' & ' num2str(err_test_2) ' & ' num2str(dof) ' & ' num2str(N_samples)]
end
%%
if 0
save("./Samples/samples_high_rank_tol_1_7.mat","Sample2",'Value_sampled','dof');
end
%%

function F = fun2(x,fun,dfun)

[m,n] = size(x);
F = zeros(m,n+1);

for i = 1:m
    y = x(i,:);
    F(i,1) = fun(y);
    for j = 1:n
        F(i,j+1) = dfun(y,j);
    end
end

end

function f = sum_gaussian(x,mu,sigma,d)

f = 0;
for i = 1:d
    f = f+exp(-sum((x-mu(i)).^2)/sigma(i)^2);
end

end
