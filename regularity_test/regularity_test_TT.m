clear
close all

addpath ..\Functions\TT
check_tt

lambda0 = 0;
lambda1 = 0.5;
lambda2 = 0;
d = 16;
x1 = 0.5;
x2 = -0.5;
fun = @(x) lambda0*sum(x.^2)+lambda1*norm(x-x1)+lambda2*sqrt(norm(x-x2));
%%
a = -1;
b = 1;
x = a:1.e-2:b;
x3 = x1;
[X,Y] = meshgrid(x,x);
F = lambda0*(X.^2+Y.^2+sum(x3^2*ones(d-2,1)))+lambda1*sqrt(((X-x1).^2)+((Y-x1).^2+sum((x3-x1)^2*ones(d-2,1))))+lambda2*sqrt(sqrt((((X-x2).^2)+((Y-x2).^2)+sum((x3-x2)^2*ones(d-2,1)))));
figure
mesh(X,Y,F)
axis([a b a b min(min(F)) max(max(F))])
xlabel('x_1')
ylabel('x_2')
title('f(x)')
view(30,20)
%%


tol = 1.e-5;
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
[V_sl,eval,Sample] = gradient_cross_new_sample(repx, repP,@(x) fun2(x,fun), tol, lambda,0,'y0',1,'kickrank',1);
V = core2cell(V_sl);
%%
dof = 0;
for i =1:d
    dof = dof+numel(V{i});
end
Sample2 = unique(Sample,'rows');
N_samples = size(Sample2,1);

%%
Value_plot = X;
x_d_2 = x3*ones(1,d-2);
for i = 1:size(X,1)
    for j = 1:size(X,2)
        Value_plot(i,j) = value_function([X(i,j) Y(i,j) x_d_2],a,b,V);
    end
end
%%
figure
mesh(X,Y,Value_plot)
axis([a b a b min(min(F)) max(max(F))])
xlabel('x_1')
ylabel('x_2')
title('TT surrogate')
view(30,20)


function F = fun2(x,fun)

[m,n] = size(x);
F = zeros(m,n+1);

for i = 1:m
    y = x(i,:);
    F(i,:) = fun(y);
end
end

