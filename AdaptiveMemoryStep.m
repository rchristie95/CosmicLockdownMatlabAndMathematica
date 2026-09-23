function [y,stats]=AdaptiveMemoryStep(rhs,y,t,last,maxstep,rtol,atol)
%ADAPTIVEMEMORYSTEP Dormand-Prince 5(4) for smooth augmented memory ODEs.
% Never use adaptive rejection on a white-noise stochastic trajectory.
step=min(maxstep,last-t); stats=[0,0]; work=0;
while t<last
    work=work+1; assert(work<1e6,'Adaptive integration work limit exceeded.');
    h=min(step,last-t); assert(t+h>t,'Adaptive step underflow.');
    k1=rhs(t,y); k2=rhs(t+h/5,y+h*k1/5);
    k3=rhs(t+3*h/10,y+h*(3*k1/40+9*k2/40));
    k4=rhs(t+4*h/5,y+h*(44*k1/45-56*k2/15+32*k3/9));
    k5=rhs(t+8*h/9,y+h*(19372*k1/6561-25360*k2/2187+64448*k3/6561-212*k4/729));
    k6=rhs(t+h,y+h*(9017*k1/3168-355*k2/33+46732*k3/5247+49*k4/176-5103*k5/18656));
    next=y+h*(35*k1/384+500*k3/1113+125*k4/192-2187*k5/6784+11*k6/84);
    k7=rhs(t+h,next);
    error=h*((35/384-5179/57600)*k1+(500/1113-7571/16695)*k3+...
        (125/192-393/640)*k4+(-2187/6784+92097/339200)*k5+(11/84-187/2100)*k6-k7/40);
    scale=atol+rtol*max(abs(y),abs(next)); err=sqrt(mean(abs(error(:)./scale(:)).^2));
    assert(isfinite(err)&&all(isfinite(next(:))),'Non-finite adaptive state.');
    if err<=1, y=next; t=t+h; stats(1)=stats(1)+1; else, stats(2)=stats(2)+1; end
    if err==0, factor=5; else, factor=min(5,max(.2,.9*err^(-.2))); end
    step=min(maxstep,h*factor);
end
end
