function y = expmv(A,v,t,tol,normBound)
%EXPMV Exponential action y=exp(t*A)*v by scaled converged Taylor steps.
% A may be sparse/dense, or a linear function handle with normBound supplied.
% This self-contained implementation avoids forming a dense exponential.
if nargin<4 || isempty(tol), tol=2e-15; end
if isa(A,'function_handle')
    assert(nargin>=5 && isfinite(normBound) && normBound>=0,'A handle requires a norm bound.');
    apply=A;
else
    assert(size(A,1)==size(A,2) && size(A,2)==size(v,1),'Incompatible exponential dimensions.');
    normBound=norm(A,1); apply=@(x) A*x;
end
assert(isscalar(t)&&isfinite(t)&&tol>0&&tol<1,'Invalid exponential settings.');
s=max(1,ceil(abs(t)*normBound/.5));
assert(isfinite(s)&&s<1e7,'Exponential work limit exceeded; reduce step or basis.');
y=v; h=t/s;
for j=1:s
    term=y; total=y; converged=false;
    for k=1:48
        term=(h/k)*apply(term); total=total+term;
        if norm(term,'fro')<=tol*max(norm(total,'fro'),1e-300)
            converged=true; break;
        end
    end
    assert(converged&&all(isfinite(total(:))),'Exponential action did not converge.');
    y=total;
end
end
