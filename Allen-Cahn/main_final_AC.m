%% Script for the resolution of the optimal control for the Allen-Cahn equation
% The surrogate model is computed via TT Cross and based on SDRE

clear
close all

addpath ..\Functions\TT
check_tt

sigma = 1.e-2;
N = 30;
x = linspace(0,1,N);
dx = x(2)-x(1);
I = eye(N);
A0 = -2*eye(N)+diag(ones(N-1,1),1)+diag(ones(N-1,1),-1);
A0(1,1) = -1; % Neumann
A0(end,end) = -1;
A = sigma*A0/dx^2;
x0 = sin(pi*x)';
gamma = dx;
Ax = @(x) A+diag(1-x.^2);
DAx = @(x,i) Derivative_AX(x,i,N);
F = @(x,u) Ax(x)*x+u;
t_final = 60;
P_sdre = @(x) P_riccati(x,Ax,dx,I,gamma);         
%%

tic
isV0 = 0; % 1 to introduce the information at the origin
tol = 1.e-4;
lambda = 0;
a = -1;
b = 1;
n = 8;
d = N;
[xx,w] = lgwt(n,a,b);
% Basis functions and theri derivatives
[P,dP] = legendre_rec(xx,-1,1,numel(xx)-1);
% P = P(2:end,2:end);
% dP = dP(2:end,2:end);
% Cell array of vectors of grid points
repx = repmat({xx},d,1);
% Cell array of basis functions and their derivatives
repP = repmat({cat(d,P,dP)},d,1);
% TT matrix of the basis functions
mkronP = mtkron([{tt_matrix(P)} repmat({tt_matrix(P)},1,d-1)]) ;
[V_sl,eval,Sample] = gradient_cross_new_sample(repx, repP, @(x) fun(x,P_sdre), tol, lambda,isV0,'y0',15,'kickrank',1);
V = core2cell(V_sl);
toc
Sample2 = unique(Sample,'rows'); % grid points sampled ( [number of points] x d )
n_sample = size(Sample2,1);
%%

Value_sampled = fun(Sample2,P_sdre);
[Vx] = value_function(Sample2,a,b,V);
norm(Vx-Value_sampled(:,1))/norm(Value_sampled(:,1))
%%
beta = 4;
ic = zeros(N,4);

for i = 1:4
    for j = 1:i
        ic(:,i) = ic(:,i)+cos(2*pi*x'*j)*j^(-beta)/2;
    end
end

err_cost = zeros(1,4);
err_test = zeros(1,4);
u = @(x) (-P_sdre(x)*x/gamma);
P = P_sdre(zeros(d,1));
K = P/gamma;
gfun = @(x,i,j) (i==j); % Actuator (here just const vector)
u_TT = @(x) multicontrolfun_leg(x',a,b,V,d,gfun,gamma,K,P,isV0)';

for i_x0 = 1:4
    x0 = ic(:,i_x0);

    Value_sampled = fun(x0',P_sdre);
    Value_sampled(1)
    [Vx] = value_function(x0',a,b,V);
    err_test(i_x0) = norm(Vx-Value_sampled(:,1))/norm(Value_sampled(:,1));

    [t_sdre, y_sdre] = ode15s(@(t,x) F(x,u(x)),[0 t_final],x0);
    nt = length(t_sdre);
    cost = zeros(nt,1);
    for i = 1:nt
        cost(i) = dx*sum(y_sdre(i,:).^2)+gamma*sum(u(y_sdre(i,:)').^2);
    end
    total_cost = sum((t_sdre(2:end)-t_sdre(1:end-1)).*(cost(2:end)+cost(1:end-1)))/2;

    [t_surr, y_surr] = ode15s(@(t,x) F(x,u_TT(x)),[0 t_final],x0);
    nt = length(t_surr);
    cost = zeros(nt,1);
    for i = 1:nt
        cost(i) = dx*sum(y_surr(i,:).^2)+gamma*sum(u(y_surr(i,:)').^2);
    end
    total_cost_surr = sum((t_surr(2:end)-t_surr(1:end-1)).*(cost(2:end)+cost(1:end-1)))/2;
    err_cost(i_x0) = abs(total_cost-total_cost_surr);

end


%%

[X,T] = meshgrid(x,t_surr);
surf(X,T,y_surr,'LineStyle','none')
xlabel('x')
ylabel('Time')
title('TT Controlled')




[t_unc, y_unc] = ode15s(@(t,x) F(x,0),[0 t_final],x0);
nt = length(t_unc);
cost = zeros(nt,1);
for i = 1:nt
    cost(i) = dx*sum(y_unc(i,:)).^2;
end
total_cost_unc = sum((t_unc(2:end)-t_unc(1:end-1)).*(cost(2:end)+cost(1:end-1)))/2

figure

[X,T] = meshgrid(x,t_unc);
surf(X,T,y_unc,'LineStyle','none')
xlabel('x')
ylabel('Time')
title('Uncontrolled')

function  P = P_riccati(x,Ax,dx,I,gamma)

P = gamma*(sqrtm(dx*I/gamma+Ax(x)*Ax(x))+Ax(x));

end

function F = fun(x,P)

[m,n] = size(x);
F = zeros(m,n+1);

for i = 1:m
    y = x(i,:);
    P0 = P(y);
    v = [y*P0*y', (2*P0*y')'];
    F(i,:) = v;
end


end