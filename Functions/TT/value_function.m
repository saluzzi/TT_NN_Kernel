% Computes the value function given a tt_tensor of value function coefficients
% in the Legendre basis
% Inputs:
%   x: a N x d matrix of coordinate positions where V(x) should be evaluated
%   a: left boundary of the domain x \in [a,b]^d
%   b: right boudary of the domain
%   V: a tt_tensor of the value function

% 
% Output:
%   ux: A N x 1 vector of control values at x

function [Vx] = value_function(x,a,b,V)
d = numel(V);
nv = size(V{1},2);
nt = size(x,1);
Vx = zeros(nt,1);
% Compute Legendre polynomials and their derivatives
[p,~] = legendre_rec(x(:), a, b, nv-1);
p = reshape(p, nt, d, nv);
% Loop over all points in x
for k=1:nt
        Vxj = 1;
        for i=1:d
            Vxj = Vxj*reshape(V{i}, size(Vxj,2), []);
            Vxj = reshape(Vxj, nv, []);
            Vxj = reshape(p(k,i,:), 1, [])*Vxj;
        end
        Vx(k) = Vx(k) + Vxj;
end
end
