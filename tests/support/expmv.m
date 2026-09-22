function result=expmv(A,v,t)
% TEST ONLY: exact dense exponential for the tiny regression-test matrices.
% Do not put tests/support on the path for production-sized calculations.
assert(size(A,1)<=64,'Test exponential is restricted to matrices of size <=64.');
result=expm(full(t*A))*v;
end
