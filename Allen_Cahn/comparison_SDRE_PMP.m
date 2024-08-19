% Comparison between PMP and SDRE for the resolution of the optimal control
% of the Allen-Cahn equation
clear
close all

addpath ..\Functions\TT
N = 30;
sigma = 1e-2;
x = linspace(0, 1, N);
dx = x(2) - x(1);
I = eye(N);

A0 = -2 * eye(N) + diag(ones(N-1, 1), 1) + diag(ones(N-1, 1), -1);
A0(1, 2) = 2; % Neumann
A0(end, end-1) = 2;
A = sparse(sigma * A0 / dx^2);

gamma = dx;
tau = 5e-2;


B = speye(N);
R = speye(N);
Q = speye(N);
Q_discr = dx * Q;
R_discr = dx * R;
T = 3;
A_tilde = speye(size(A)) - tau * A;
A_tilde_T = (speye(size(A)) - tau * A'); % used for semiimplicit backwards solution
step_size = 0.005;


x0 = sin(pi * x)';


%%
y = [0;x0];
t_steps = y(1):tau:T;
u0 = zeros(length(x0),length(y(1):tau:T));
fun = @(x,u) cost_gradient(u,x(2:end),x(1),R_discr,B,tau,A_tilde_T,Q_discr,A_tilde,T);
options = optimoptions('fminunc','SpecifyObjectiveGradient',true,'Display','off');
tic
[Unew, ~] = fminunc(@(u) fun(y,u), u0, options);
x_opt = solve_fixed_u(x0, Unew,t_steps,tau,A_tilde);
total_cost_PMP = compute_costs(x_opt, Unew,t_steps,Q_discr,R_discr);
time_PMP = toc;
fprintf('\n PMP: Total cost: %.2e \t  Time: %.2e \n\n', total_cost_PMP, time_PMP)
%%



Ax = @(x) A+diag(1-x.^2);


BB = B*inv(R_discr)*B';
tol_trunc = 1e-5;
linesearch = 1;
it_linesearch = 1;
outer_it = 30;
normQ = norm(Q,'fro');
res = @(A,X) norm(A*X+X*A'-X*BB*X+Q_discr,'fro')/normQ;
P_sdre = @(x) newton_kleinman(Ax(x),BB,Q_discr,speye(N),tol_trunc,outer_it,res,linesearch,it_linesearch);

u = @(x) (-P_sdre(x)*x/gamma);
F = @(x,u) Ax(x)*x+u;
nt = length(t_steps);
y_sdre = zeros(N,nt);
u_sdre = zeros(N,nt);
xnew = x0;
y_sdre(:,1) = x0;
tic
for i = 1:nt-1
    u_sdre(:,i) = u(xnew);
    xnew = A_tilde\(xnew+tau*(xnew - xnew.^3+u_sdre(:,i)));
    y_sdre(:,i+1) = xnew;
end
t_sdre = t_steps';
y_sdre = y_sdre';
nt = length(t_sdre);
cost = zeros(nt,1);
for i = 1:nt
    cost(i) = dx*sum(y_sdre(i,:).^2)+gamma*sum(u_sdre(:,i)'.^2);
end
total_cost_SDRE = sum((t_sdre(2:end)-t_sdre(1:end-1)).*(cost(2:end)+cost(1:end-1)))/2;
time_SDRE = toc;

fprintf('SDRE: Total cost: %.2e \t  Time: %.2e \n', total_cost_SDRE, time_SDRE)



%%


function [J,DJ] = cost_gradient(u,x,t,R_discr,B,tau,A_tilde_T,Q_discr,A_tilde,T)

t_steps = t:tau:T;
x_opt = solve_fixed_u(x, u,t_steps,tau,A_tilde);
J = compute_costs(x_opt, u,t_steps,Q_discr,R_discr);
DJ = gradJp(x, u,R_discr,B,t_steps,tau,A_tilde_T,Q_discr,A_tilde);

end


function cost = compute_costs(x, u,t_steps,Q_discr,R_discr)
    cost_vec = zeros(1, length(t_steps));
    for i0 = 1:length(t_steps)
        xcurr = x(:, i0);
        ucurr = u(:,i0);
        cost_vec(i0) = xcurr'*Q_discr*xcurr+ucurr'*R_discr*ucurr;
    end
    cost = trapz(t_steps, cost_vec);
end

function x = solve_fixed_u(x0, u,t_steps,tau,A_tilde)
x = zeros(length(x0), length(t_steps));
x(:, 1) = x0;
xold = x0;
for i0 = 1:length(t_steps) - 1
    xold = A_tilde\(xold+tau*(xold - xold.^3+u(:,i0)));
    x(:, i0 + 1) = xold;
end
end

function grad = gradJp(x0, u,R_discr,B,t_steps,tau,A_tilde_T,Q_discr,A_tilde)
    ygrad = solve_fixed_u(x0, u,t_steps,tau,A_tilde);
    p = solve_peq(ygrad, u,A_tilde_T,tau,Q_discr);
    grad = 2 * R_discr * u + B' * p;
end

function p = solve_peq(x, u,A_tilde_T,tau,Q_discr)
p = zeros(size(x, 1), size(u, 2));
p_next = 2 * Q_discr * x(:,size(p, 2));
p(:,end) = p_next;
    for i0 = size(p, 2)-1:-1:1
         p_next = A_tilde_T\(p_next + tau * ((-3 *  x(:, i0).^2 + 1) .* p_next + 2 * Q_discr * x(:, i0)));
        p(:,i0) = p_next;
    end
end

