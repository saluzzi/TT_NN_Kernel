clc
clear
close all

addpath ..\Functions\TT
check_tt

N = 30;
sigma = 1e-2;
x = linspace(0, 1, N);
dx = x(2) - x(1);
A0 = -2 * eye(N) + diag(ones(N-1, 1), 1) + diag(ones(N-1, 1), -1);
A0(1, 2) = 2; % Neumann
A0(end, end-1) = 2;
A = sparse(sigma * A0 / dx^2);
B = speye(N);
T = 1;
Ptf = dx*speye(N); Rxx = Ptf; Ruu = Ptf;
BB = B*Ruu^(-1)*B';

dot_P = @(t,y) doty(t,y,A,BB,Rxx);


%%

P0 = @(t) riccati(t,dot_P,Ptf,T);
fun = @(x) x(2:end)*P0(x(1))*x(2:end)';
tic
fun(0.5*ones(31,1)')
toc
%%

tol = 1.e-5;
lambda = 0;
n = 3;
d = N+1;
a = 0;
b = 1;
[x,~] = lgwt(n,a,b);
% Basis functions and theri derivatives
[P,dP] = legendre_rec(x,a,b,numel(x)-1);
% Cell array of vectors of grid points
repx = repmat({x},d,1);
% Cell array of basis functions and their derivatives
repP = repmat({cat(d,P,dP)},d,1);
% TT matrix of the basis functions
mkronP = mtkron([{tt_matrix(P)} repmat({tt_matrix(P)},1,d-1)]) ;
tic
[V_sl,eval,Sample] = gradient_cross_new_sample(repx, repP,@(x) fun2(x,fun), tol, lambda,'y0',1,'kickrank',1);
toc
V = core2cell(V_sl);
%%
dof = 0;
for i = 1:d
    dof = dof+numel(V{i});
end
Sample2 = unique(Sample,'rows');
N_samples = size(Sample2,1);
%%

Value_sampled = fun2(Sample2,fun);
[Vx] = value_function(Sample2,a,b,V);
err_train_inf = max(abs(Vx-Value_sampled(:,1)));
norm(Vx-Value_sampled(:,1))/norm(Value_sampled(:,1))
%%
X_test = rand(5000,d);
Value_test = fun2(X_test,fun);
[Vx_test] = value_function(X_test,a,b,V);
err_test_inf = max(abs(Vx_test-Value_test(:,1)));
norm(Vx_test-Value_test(:,1))/norm(Value_test(:,1))


function P0 = riccati(t,dot_P,Ptf,T)
if abs(t-1)<= 1.e-7
    P0 = unvec(y);
else
    [~,y] = ode45(dot_P,[t 1.],vec(Ptf));
    P0 =  unvec(y(1,:));
end
end

function F = fun2(x,fun)

[m,n] = size(x);
F = zeros(m,n+1);

for i = 1:m
    y = x(i,:);
    F(i,:) = fun(y);
end
end


function [doy] = doty(t,y,A,BB,Rx)
P = unvec(y);
dotP = P*A+A'*P+Rx-P*BB*P;
doy = vec(dotP);
end

function P=unvec(y)
N = max(roots([1 1 -2*length(y)]));
P=[];kk=N;kk0=1;
for ii = 1:N
    P(ii,ii:N) = [y(kk0+[0:kk-1])]';
    kk0 = kk0+kk;
    kk = kk-1;
end
P = (P+P')-diag(diag(P));
end

function y=vec(P)
y=[];
for ii=1:length(P)
    y=[y;P(ii,ii:end)'];
end
end



