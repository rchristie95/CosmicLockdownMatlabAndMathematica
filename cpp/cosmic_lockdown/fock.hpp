#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace fock {
using C=std::complex<double>;
using M=Eigen::MatrixXcd;
using V=Eigen::VectorXcd;
using R=Eigen::VectorXd;
constexpr double pi=3.14159265358979323846;
const C I(0,1);
inline void check(bool ok,const std::string& s){if(!ok)throw std::runtime_error(s);}
struct Config {
    std::string workflow="sse",model="x";
    int basis=24,frames=21,grid=65;
    double hbar=1,mu=.5,beta3=.025,beta4=.13,lambda=.05,H=5;
    double initial=-2,final=1,step=.001,volume=0,omega=1,extent=10;
    unsigned long long seed=20260928;
    std::vector<double> times,noise,steps;
    std::vector<C> zeta;
    double rtol=1e-9,atol=1e-11;
    bool wigner=true;
    int power()const{return model=="x2"?2:model=="x3"?3:1;}
    bool nm()const{return workflow.rfind("nm-",0)==0;}
    double vol()const{return volume>0?volume:(model=="x"&&!nm()?4*std::sqrt(2.):1.);}
    double rate(double t)const{
        double pref=power()==1?131*pi/(256*std::pow(mu,5)*vol()):
            power()==2?pi/(64*std::pow(mu,4)):pi/(8*std::pow(mu,3));
        return pref*lambda*lambda*std::exp(6*t);
    }
};
struct Ops {
    M x,p,p2,potential,coupling;
    explicit Ops(int n,const Config& c){
        M a=M::Zero(n,n);for(int j=1;j<n;++j)a(j-1,j)=std::sqrt(double(j));
        x=std::sqrt(c.hbar/2)*(a+a.adjoint());p=I*std::sqrt(c.hbar/2)*(a.adjoint()-a);
        M x2=x*x,x3=x2*x;p2=p*p;
        potential=-c.mu*c.mu*x2/2+(2*c.beta3*c.mu/3)*x3+
            ((c.beta4*c.beta4-c.beta3*c.beta3)/4)*(x2*x2);
        coupling=c.power()==1?x:c.power()==2?x2:x3;
    }
    M H(double t,const Config& c)const{
        return std::exp(-3*t)*p2/(2*c.H*c.vol())+std::exp(3*t)*c.vol()*potential/c.H;
    }
};
inline V normalized(const V& v){check(v.allFinite()&&v.norm()>0,"Invalid state");return v/v.norm();}
inline M density(const V& v){return v*v.adjoint();}
inline M comm(const M& a,const M& b){return a*b-b*a;}
inline V ground(const M& h){
    Eigen::SelfAdjointEigenSolver<M> e(h);check(e.info()==Eigen::Success,"Ground eigensolver failed");
    V v=e.eigenvectors().col(0);Eigen::Index j;v.cwiseAbs().maxCoeff(&j);
    v*=std::conj(v(j))/std::abs(v(j));return v;
}
// Exact Gaussian instrument for a Hermitian observable in its eigenbasis.
// A single normal quantile samples the Born-weighted Gaussian mixture. This
// retains a scalar innovation for externally supplied/coupled noise paths.
inline V measurement(V v,const R& a,double q,double z){
    if(q==0)return v;
    check(q>0&&std::isfinite(q)&&std::isfinite(z),"Invalid measurement clock/noise");
    R weights=v.cwiseAbs2();weights/=weights.sum();
    R centers=2*std::sqrt(q)*a;
    double lo=z+centers.minCoeff(),hi=z+centers.maxCoeff();
    // Survival probabilities avoid cancellation for positive normal quantiles.
    bool upper=z>0;double target=.5*std::erfc((upper?z:-z)/std::sqrt(2.));
    for(int k=0;k<80;++k){
        double mid=.5*lo+.5*hi,total=0;
        for(int j=0;j<a.size();++j)total+=weights(j)*.5*std::erfc((upper?mid-centers(j):centers(j)-mid)/std::sqrt(2.));
        if((upper&&total>target)||(!upper&&total<target))lo=mid;else hi=mid;
        if(hi-lo<8*std::numeric_limits<double>::epsilon()*std::max(1.,std::abs(mid)))break;
    }
    double y=std::sqrt(q)*(.5*lo+.5*hi);
    R logs(a.size());double peak=-std::numeric_limits<double>::infinity();
    for(int j=0;j<a.size();++j){logs(j)=std::abs(v(j))>0?std::log(std::abs(v(j)))+a(j)*y-q*a(j)*a(j):-std::numeric_limits<double>::infinity();peak=std::max(peak,logs(j));}
    for(int j=0;j<a.size();++j)v(j)=std::abs(v(j))>0?v(j)/std::abs(v(j))*std::exp(logs(j)-peak):C(0);
    return normalized(v);
}
// Cached spectral Strang flow: kinetic energy, commuting potential/measurement,
// kinetic energy. Every factor is unitary or a completely positive instrument.
struct Spectral {
    M ux,up;R x,p2,potential,a;
    Spectral(const Ops& o,const Config& c){
        Eigen::SelfAdjointEigenSolver<M> ex(o.x),ep(o.p2);
        check(ex.info()==Eigen::Success&&ep.info()==Eigen::Success,"Spectral decomposition failed");
        ux=ex.eigenvectors();up=ep.eigenvectors();x=ex.eigenvalues();p2=ep.eigenvalues();
        potential=(-c.mu*c.mu*x.array().square()/2+2*c.beta3*c.mu*x.array().cube()/3+
            (c.beta4*c.beta4-c.beta3*c.beta3)*x.array().pow(4)/4).matrix();
        a=x.array().pow(c.power()).matrix();
    }
    static V phase(const R& values,double clock){return (-I*clock*values.cast<C>().array()).exp().matrix();}
    V kinetic(const V& v,double clock)const{return up*(phase(p2,clock).array()*(up.adjoint()*v).array()).matrix();}
    M kinetic_density(const M& r,double clock)const{
        V d=phase(p2,clock);M z=up.adjoint()*r*up;z=d.asDiagonal()*z*d.conjugate().asDiagonal();return up*z*up.adjoint();
    }
    V state(const V& v,double t,double dt,const Config& c,double q=0,double z=0)const{
        double kt=std::exp(-3*t)*(-std::expm1(-3*dt))/(6*c.H*c.vol()*c.hbar);
        double vt=std::exp(3*t)*std::expm1(3*dt)*c.vol()/(3*c.H*c.hbar);
        V u=ux.adjoint()*kinetic(v,.5*kt);
        if(q>0)u=measurement(u,a,q,z);
        u.array()*=phase(potential,vt).array();return kinetic(V(ux*u),.5*kt);
    }
    V closed(V v,double t,double dt,const Config& c)const{
        // Fourth-order symmetric composition; negative substeps only for unitary evolution.
        const double w=1/(2-std::cbrt(2.));
        for(double coefficient:{w,1-2*w,w}){double h=coefficient*dt;v=state(v,t,h,c);t+=h;}return v;
    }
    M channel(const M& r,double t,double dt,const Config& c,double q)const{
        double kt=std::exp(-3*t)*(-std::expm1(-3*dt))/(6*c.H*c.vol()*c.hbar);
        double vt=std::exp(3*t)*std::expm1(3*dt)*c.vol()/(3*c.H*c.hbar);
        M u=ux.adjoint()*kinetic_density(r,.5*kt)*ux;
        for(int j=0;j<u.cols();++j)for(int i=0;i<u.rows();++i)
            u(i,j)*=std::exp(-I*vt*(potential(i)-potential(j))-.5*q*std::pow(a(i)-a(j),2));
        return kinetic_density(M(ux*u*ux.adjoint()),.5*kt);
    }
};
struct AdaptiveStats {unsigned long long accepted=0,rejected=0;};
// Dormand-Prince 5(4), with a componentwise weighted RMS error. Only smooth
// coloured-noise augmented ODEs use adaptivity; never reject white-noise steps.
inline M adaptive(const std::function<M(double,const M&)>& rhs,M y,double t,double end,
                  double maxstep,double rtol,double atol,AdaptiveStats& stats){
    double step=std::min(maxstep,end-t);unsigned work=0;
    while(t<end){
        check(++work<1000000,"Adaptive integration work limit exceeded");
        double h=std::min(step,end-t);check(t+h>t,"Adaptive step underflow");
        M k1=rhs(t,y),k2=rhs(t+h/5,y+h*k1/5);
        M k3=rhs(t+3*h/10,y+h*(3*k1/40+9*k2/40));
        M k4=rhs(t+4*h/5,y+h*(44*k1/45-56*k2/15+32*k3/9));
        M k5=rhs(t+8*h/9,y+h*(19372*k1/6561-25360*k2/2187+64448*k3/6561-212*k4/729));
        M k6=rhs(t+h,y+h*(9017*k1/3168-355*k2/33+46732*k3/5247+49*k4/176-5103*k5/18656));
        M next=y+h*(35*k1/384+500*k3/1113+125*k4/192-2187*k5/6784+11*k6/84);
        M k7=rhs(t+h,next);
        M error=h*((35./384-5179./57600)*k1+(500./1113-7571./16695)*k3+
            (125./192-393./640)*k4+(-2187./6784+92097./339200)*k5+(11./84-187./2100)*k6-k7/40);
        double err=0;
        for(int j=0;j<y.size();++j){double scale=atol+rtol*std::max(std::abs(y(j)),std::abs(next(j)));err+=std::norm(error(j)/scale);}
        err=std::sqrt(err/y.size());check(std::isfinite(err)&&next.allFinite(),"Non-finite adaptive state");
        if(err<=1){y=std::move(next);t+=h;++stats.accepted;}else ++stats.rejected;
        step=std::min(maxstep,h*(err==0?5:std::clamp(.9*std::pow(err,-.2),.2,5.)));
    }
    return y;
}
inline R bath(double t,const Config& c){
    double d=128*std::pow(c.H*c.omega,3);R v(4);
    v<<std::sqrt(27/d)*std::cos(c.omega*t),std::sqrt(27/d)*std::sin(c.omega*t),
       std::sqrt(3/d)*std::cos(3*c.omega*t),std::sqrt(3/d)*std::sin(3*c.omega*t);return v;
}
struct Snapshot {double t;V psi;M rho;};
struct Result {std::vector<Snapshot> frames;std::vector<double> noise;std::vector<C> zeta;double residual=0;AdaptiveStats stats;};
inline Result solve(Config c){
    check(c.basis>=2&&c.basis<=512&&c.step>0&&c.final>=c.initial&&c.H>0&&c.hbar>0&&c.mu>0&&
          c.rtol>0&&c.atol>0&&c.lambda>=0&&c.omega>0&&c.volume>=0&&c.vol()>0,"Invalid physical/numerical parameters");
    for(double v:{c.hbar,c.mu,c.beta3,c.beta4,c.lambda,c.H,c.initial,c.final,c.step,c.volume,c.omega,c.rtol,c.atol})
        check(std::isfinite(v),"Non-finite parameter");
    check(c.model=="x"||c.model=="x2"||c.model=="x3","Unknown model");
    check(!(c.model=="x"&&(c.workflow=="sse"||c.workflow=="lindblad"))||c.hbar==1,"Paper X model requires hbar=1");
    bool sse=c.workflow=="sse",lind=c.workflow=="lindblad",closed=c.workflow=="closed";
    bool adi=c.workflow=="adiabatic",nms=c.workflow=="nm-sse";
    bool nmd=c.workflow=="nm-density";
    check(sse||lind||closed||adi||nms||nmd,"Unknown workflow");
    check(!c.nm()||c.model=="x","Non-Markovian workflows use X coupling");
    int multiplier=(sse||adi||c.nm())?2:3;
    Ops small(c.basis,c),big(multiplier*c.basis,c);M h0=big.H(c.initial,c);
    V psi=ground(h0);Result out; Spectral sb(big,c),ss(small,c);
    double e=(psi.dot(h0*psi)).real();out.residual=(h0*psi-e*psi).norm()/std::max(1.,h0.norm());
    M rho=density(normalized(V(psi.head(c.basis))));
    std::mt19937_64 rng(c.seed);std::normal_distribution<double> normal;
    if(c.zeta.empty())for(int k=0;k<4;++k)c.zeta.push_back(C(normal(rng),normal(rng))/std::sqrt(2.));
    check(c.zeta.size()==4,"Need four complex bath coefficients");out.zeta=c.zeta;
    std::vector<double> times=c.times;
    if(times.empty()){
        check(c.frames>=2,"Need at least two output frames");
        for(int j=0;j<c.frames;++j)times.push_back(c.initial+(c.final-c.initial)*j/(c.frames-1));
        if(c.initial==c.final)times.resize(1);
    }
    check(!times.empty()&&std::abs(times.front()-c.initial)<1e-12&&std::abs(times.back()-c.final)<1e-12,"Output times must include both endpoints");
    for(size_t k=1;k<times.size();++k)check(times[k]>times[k-1],"Output times must increase");
    check(c.steps.empty()||c.steps.size()==times.size()-1,"Need one maximum step per output interval");
    M aux=M::Zero(c.basis,4),memory=M::Zero(c.basis,5*c.basis);
    bool started=false;size_t ni=0;
    auto store=[&](double t){
        V p;M r;
        if(adi){p=normalized(V(ground(big.H(t,c)).head(c.basis)));r=density(p);}
        else if(lind||nmd){r=rho;}
        else {p=psi.head(c.basis);if(!closed)p=normalized(p);r=density(p);}
        check(r.allFinite(),"Non-finite output");out.frames.push_back({t,p,r});
    };
    if(nms&&c.initial>=-1)psi=normalized(V(psi.head(c.basis)));
    double t=c.initial;store(t);
    for(size_t frame=1;frame<times.size();++frame){
        while(!adi&&t<times[frame]-1e-13){
            double maxstep=c.steps.empty()?c.step:c.steps[frame-1];
            check(maxstep>0&&std::isfinite(maxstep),"Invalid interval step");
            double dt=(nmd||(nms&&t>=-1))?times[frame]-t:std::min(maxstep,times[frame]-t);
            bool switched_model=(sse||lind)&&c.power()!=1;
            if((switched_model||nms)&&t<-1&&t+dt>-1)dt=-1-t;
            double next=t+dt;
            check(next>t,"Step is too small to advance floating-point time");
            if(closed)psi=sb.closed(psi,t,dt,c);
            else if(sse){
                double z;
                if(c.noise.empty())z=normal(rng);else{check(ni<c.noise.size(),"Noise file too short");z=c.noise[ni];}
                ++ni;out.noise.push_back(z);
                double q=(c.power()!=1&&t<-1)?0:c.rate(t)*std::expm1(6*dt)/(6*c.hbar);
                psi=sb.state(psi,t,dt,c,q,z);
            }else if(lind){
                if(c.power()!=1&&t<-1){psi=sb.closed(psi,t,dt,c);rho=density(normalized(V(psi.head(c.basis))));}
                else rho=ss.channel(rho,t,dt,c,c.rate(t)*std::expm1(6*dt)/(6*c.hbar));
            }else if(nms){
                if(t<-1){psi=sb.closed(psi,t,dt,c);}
                else{
                    if(!started){psi=normalized(V(psi.head(c.basis)));started=true;}
                    M state(c.basis,5);state.col(0)=psi;state.rightCols(4)=aux;
                    auto rhs=[&](double u,const M& y)->M{
                        V p=y.col(0);R row=bath(u,c);C eta=0;for(int j=0;j<4;++j)eta+=row(j)*c.zeta[j];
                        M l=(c.lambda/c.H)*std::exp(3*u)*small.x;
                        double mean=(p.dot(l*p)).real()/p.squaredNorm();M centered=l-mean*M::Identity(c.basis,c.basis);
                        M d=(-I/c.hbar)*small.H(u,c)+std::conj(eta)*centered;
                        V f=d*p-centered*(y.rightCols(4)*row.cast<C>());
                        // Continuous normalized gauge; no finite-step norm repair.
                        f-=p*(p.dot(f).real()/p.squaredNorm());
                        M dy(c.basis,5);dy.col(0)=f;dy.rightCols(4)=d*y.rightCols(4)+(centered*p)*row.transpose();return dy;
                    };
                    state=adaptive(rhs,state,t,next,maxstep,c.rtol,c.atol,out.stats);
                    psi=state.col(0);aux=state.rightCols(4);
                }
            }else if(nmd){
                memory.leftCols(c.basis)=rho;
                auto rhs=[&](double u,const M& y)->M{
                    M l=(c.lambda/c.H)*std::exp(3*u)*small.x;R row=bath(u,c);
                    M r=y.leftCols(c.basis),k=comm(l,r),sum=M::Zero(c.basis,c.basis),dy(y.rows(),y.cols());
                    for(int j=0;j<4;++j){sum+=row(j)*y.middleCols((j+1)*c.basis,c.basis);dy.middleCols((j+1)*c.basis,c.basis)=row(j)*k;}
                    dy.leftCols(c.basis)=(-I/c.hbar)*comm(small.H(u,c),r)-comm(l,sum);return dy;
                };
                memory=adaptive(rhs,memory,t,next,maxstep,c.rtol,c.atol,out.stats);rho=memory.leftCols(c.basis);
            }

            t=next;
        }
        t=times[frame];store(t);
    }
    if(sse&&!c.noise.empty())check(ni==c.noise.size(),"Noise file has unused samples");
    return out;
}
// Analytic harmonic-oscillator Wigner basis, retaining negative eigenvalues.
inline double wigner(const M& rho,double x,double p,double hbar){
    double rr=(x*x+p*p)/hbar,value=0;
    // Normalized associated-Laguerre recurrence with logarithmic scaling:
    // avoids factorial/power overflow at the original basis sizes (~400).
    for(int d=0;d<rho.rows();++d){
        if(rr==0&&d>0)continue;
        double logscale=-rr+(d?d*.5*std::log(2*rr):0)-.5*std::lgamma(d+1.);
        C phase=std::polar(1.,d*std::atan2(-p,x));
        double previous=0,current=1;
        for(int n=0;n+d<rho.rows();++n){
            double amplitude=current==0?0:std::copysign(std::exp(logscale+std::log(std::abs(current))),current);
            value+=(d?2*std::real(rho(n+d,n)*phase):rho(n,n).real())*amplitude;
            double denominator=std::sqrt(double((n+1)*(n+d+1)));
            double next=-(2*n+1+d-2*rr)*current/denominator-
                std::sqrt(double(n*(n+d)))*previous/denominator;
            previous=current;current=next;
            double scale=std::max(std::abs(previous),std::abs(current));
            if(scale>1e100){previous*=1e-100;current*=1e-100;logscale+=100*std::log(10.);}
            else if(scale>0&&scale<1e-100){previous*=1e100;current*=1e100;logscale-=100*std::log(10.);}
        }
    }
    check(std::isfinite(value),"Wigner recurrence failed");return value/(pi*hbar);
}
}
