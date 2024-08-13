function [X, Cpu,it, resvec] = newton_kleinman(A,B,C,Xguess,tol_trunc,outer_it,res,linesearch,it_line_search)

% Compute the solution of the CARE A X + X A' - X B X + C = 0 with Newton Kleinman method
%
%  Output:
%
%  X   Solution of the CARE
%  Cpu  CPU time
%------------------------------------------------------------------------------
%residual_old = 1+tol_trunc;
it = 0;
X = Xguess;
err_old = res(A,X);

if linesearch
    R = @(A,X) A*X+X*A'-X*B*X+C;
    fmin = @(t,a,b,c) a*(1-t)^2-2*b*(1-t)*t^2+c*t^4;
end

tic
while err_old > max([tol_trunc 1.e-10]) && it<outer_it
    it = it+1;
    A1 = A-X*(B);
    %l = eig(A1);
    %max_eigA1 = max(l);
    %min_eigA1 = min(l);
    QQ = X*(B)*X+C;
    try
        Xnew = lyap(A1,QQ);
    catch
        keyboard
    end
    % X = tril(triu(X, -ndiag), ndiag);
    if linesearch && it <= it_line_search
        Z = Xnew-X;
        V = Z*B*Z;
        Rx = R(A,X);
        alfa = trace(Rx*Rx);
        beta = trace(Rx*V);
        gamma = trace(V*V);
        fmint = @(t) fmin(t,alfa,beta,gamma);
        tbar = fminbnd(fmint,0,2);
        X = X+tbar*Z;
    else
        X = Xnew;
    end
    err_new = res(A,X);
    resvec(it) = err_new;
    err_old = err_new;
    Cpu(it) = toc;
    %fprintf('Newton-Kleinmann:: It = %d,\t Residual: %.2e \t  Time: %.2f s \n', it, err_new,Cpu(it))
end

end
