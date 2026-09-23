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
inline double bound(const M& m){return m.cwiseAbs().colwise().sum().maxCoeff();}
// Scaled Taylor exponential action. No dense n^2-by-n^2 Liouvillian is formed.
inline M exp_action(const std::function<M(const M&)>& apply,const M& v,double dt,double norm_bound){
    const double count=std::ceil(std::abs(dt)*norm_bound/.5);
    check(std::isfinite(count)&&count<1e7,"Exponential action exceeds work limit; reduce step/basis");
    int pieces=std::max(1,int(count));double h=dt/pieces;M out=v;
    for(int s=0;s<pieces;++s){
        M sum=out,term=out;bool converged=false;
        for(int k=1;k<=48;++k){
            term=(h/k)*apply(term);sum+=term;
            if(term.norm()<=2e-15*std::max(sum.norm(),1e-300)){converged=true;break;}
        }
        check(converged&&sum.allFinite(),"Exponential action did not converge");out=std::move(sum);
    }
    return out;
}
inline V unitary(const V& psi,const M& h,double dt,double hbar){
    return normalized(exp_action([&](const M& a)->M{return (-I/hbar)*(h*a);},psi,dt,bound(h)/hbar));
}
inline M gkls(const M& rho,const M& h,const M& l,double dt,double hbar){
    M l2=l.adjoint()*l;
    auto apply=[&](const M& r)->M{return (-I/hbar)*comm(h,r)+(l*r*l.adjoint()-.5*(l2*r+r*l2))/hbar;};
    M r=exp_action(apply,rho,dt,(2*bound(h)+2*bound(l)*bound(l))/hbar);
    check(r.allFinite()&&r.trace().real()>0,"Invalid GKLS density");return r/r.trace().real();
}
inline V diffusion(const V& v,const M& l,double hbar){
    // The normalized expectation extends the SSE to intermediate SRK stages.
    double mean=(v.dot(l*v)).real()/v.squaredNorm();return (l*v-mean*v)/std::sqrt(hbar);
}
inline V drift(const V& v,const M& h,const M& l,double hbar){
    double mean=(v.dot(l*v)).real()/v.squaredNorm();V w=l*v-mean*v;
    return (-I/hbar)*(h*v)-.5/hbar*(l*w-mean*w);
}
inline V sse_step(const V& in,const M& h0,const M& h1,const M& l0,const M& l1,
                  double dt,double dw,double hbar,int power){
    V v=normalized(in),f=drift(v,h0,l0,hbar),g=diffusion(v,l0,hbar);
    if(power!=2)return normalized(v+dt*f+dw*g);
    double sq=std::sqrt(dt),i11=.5*(dw*dw-dt);
    V pred=v+dt*f,shift=g*(i11/sq);
    return normalized(v+.5*dt*(f+drift(pred,h1,l1,hbar))+dw*g+
        .5*sq*(diffusion(v+shift,l1,hbar)-diffusion(v-shift,l1,hbar)));
}
inline R bath(double t,const Config& c){
    double d=128*std::pow(c.H*c.omega,3);R v(4);
    v<<std::sqrt(27/d)*std::cos(c.omega*t),std::sqrt(27/d)*std::sin(c.omega*t),
       std::sqrt(3/d)*std::cos(3*c.omega*t),std::sqrt(3/d)*std::sin(3*c.omega*t);return v;
}
struct Snapshot {double t;V psi;M rho;};
struct Result {std::vector<Snapshot> frames;std::vector<double> noise;std::vector<C> zeta;double residual=0;};
inline Result solve(Config c){
    check(c.basis>=2&&c.basis<=512&&c.step>0&&c.final>=c.initial&&c.H>0&&c.hbar>0&&c.mu>0&&
          c.lambda>=0&&c.omega>0&&c.vol()>0,"Invalid physical/numerical parameters");
    check(c.model=="x"||c.model=="x2"||c.model=="x3","Unknown model");
    check(!(c.model=="x"&&(c.workflow=="sse"||c.workflow=="lindblad"))||c.hbar==1,"Paper X model requires hbar=1");
    bool sse=c.workflow=="sse",lind=c.workflow=="lindblad",closed=c.workflow=="closed";
    bool adi=c.workflow=="adiabatic",nms=c.workflow=="nm-sse";
    bool nmd=c.workflow=="nm-density"||c.workflow=="nm-density-full";
    check(sse||lind||closed||adi||nms||nmd,"Unknown workflow");
    check(!c.nm()||c.model=="x","Non-Markovian workflows use X coupling");
    int multiplier=(sse||adi||c.nm())?2:3;
    Ops small(c.basis,c),big(multiplier*c.basis,c);M h0=big.H(c.initial,c);
    V psi=ground(h0);Result out;
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
    M aux=M::Zero(c.basis,4);std::vector<M> accum(4,M::Zero(c.basis,c.basis)),hist;
    std::vector<R> history_weights;bool started=false;size_t ni=0;
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
            double dt=std::min(maxstep,times[frame]-t);
            bool switched_model=(sse||lind)&&c.power()!=1;
            if((switched_model||nms)&&t<-1&&t+dt>-1)dt=-1-t;
            double next=t+dt;
            if(closed)psi=unitary(psi,big.H(t+.5*dt,c),dt,c.hbar);
            else if(sse){
                double z;
                if(c.noise.empty())z=normal(rng);else{check(ni<c.noise.size(),"Noise file too short");z=c.noise[ni];}
                ++ni;out.noise.push_back(z);
                if(c.lambda==0||(c.power()!=1&&t<-1))psi=unitary(psi,big.H(t,c),dt,c.hbar);
                else psi=sse_step(psi,big.H(t,c),big.H(next,c),std::sqrt(c.rate(t))*big.coupling,
                    std::sqrt(c.rate(next))*big.coupling,dt,std::sqrt(dt)*z,c.hbar,c.power());
            }else if(lind){
                if(c.power()!=1&&t<-1){psi=unitary(psi,big.H(t+.5*dt,c),dt,c.hbar);rho=density(normalized(V(psi.head(c.basis))));}
                else rho=gkls(rho,small.H(t+.5*dt,c),std::sqrt(c.rate(t+.5*dt))*small.coupling,dt,c.hbar);
            }else if(nms){
                if(t<-1){psi=unitary(psi,big.H(t,c),dt,c.hbar);}
                else{
                    if(!started){
                        psi=normalized(V(psi.head(c.basis)));M l=(c.lambda/c.H)*std::exp(3*t)*small.x;
                        double mean=(psi.dot(l*psi)).real();V f=(l*psi-mean*psi)*dt;
                        aux=f*bath(t,c).cast<C>().transpose();started=true;
                    }
                    M ln=(c.lambda/c.H)*std::exp(3*t)*small.x,le=(c.lambda/c.H)*std::exp(3*next)*small.x;
                    R row=bath(next,c);C eta=0;for(int j=0;j<4;++j)eta+=row(j)*c.zeta[j];
                    double mean=(psi.dot(ln*psi)).real();M centered=ln-mean*M::Identity(c.basis,c.basis);
                    M dn=(-I/c.hbar)*small.H(t,c)+std::conj(eta)*centered;
                    V drift0=dn*psi-centered*(aux*row.cast<C>());
                    V pred=normalized(psi+dt*drift0);M auxp=aux+dt*(dn*aux);
                    double mp=(pred.dot(le*pred)).real();M ce=le-mp*M::Identity(c.basis,c.basis);
                    M de=(-I/c.hbar)*small.H(next,c)+std::conj(eta)*ce;
                    V drift1=de*pred-ce*(auxp*row.cast<C>());
                    psi=normalized(psi+.5*dt*(drift0+drift1));
                    M homogeneous=aux+.5*dt*(dn*aux+de*auxp);
                    double me=(psi.dot(le*psi)).real();V f=(le*psi-me*psi)*dt;
                    aux=homogeneous+f*row.cast<C>().transpose();
                }
            }else if(nmd){
                M ln=(c.lambda/c.H)*std::exp(3*t)*small.x,lm=(c.lambda/c.H)*std::exp(3*(t+.5*dt))*small.x;
                M k=comm(ln,rho),sum=M::Zero(c.basis,c.basis);R past=bath(t,c),row=bath(next,c);
                if(c.workflow=="nm-density-full"){
                    hist.push_back(dt*k);history_weights.push_back(past);
                    for(size_t j=0;j<hist.size();++j)sum+=row.dot(history_weights[j])*hist[j];
                }else for(int j=0;j<4;++j){accum[j]+=dt*past(j)*k;sum+=row(j)*accum[j];}
                M d0=(-I/c.hbar)*comm(small.H(t,c),rho)-comm(ln,sum);
                M half=rho+.5*dt*d0;
                rho+=dt*((-I/c.hbar)*comm(small.H(t+.5*dt,c),half)-comm(lm,sum));
                rho=(.5*(rho+rho.adjoint())).eval();check(std::abs(rho.trace())>1e-15,"Invalid memory density trace");rho/=rho.trace();
            }
            t=next;
        }
        t=times[frame];store(t);
    }
    if(sse&&!c.noise.empty())check(ni==c.noise.size(),"Noise file has unused samples");
    return out;
}
// Analytic harmonic-oscillator Wigner basis, retaining negative eigenvalues.
inline double laguerre(int n,int a,double x){
    if(n==0)return 1;
    double old=1,v=1+a-x;
    for(int k=1;k<n;++k){double next=((2*k+1+a-x)*v-(k+a)*old)/(k+1);old=v;v=next;}return v;
}
inline double wigner(const M& rho,double x,double p,double hbar){
    double rr=(x*x+p*p)/hbar,value=0;C z=std::sqrt(2/hbar)*C(x,-p);
    for(int n=0;n<rho.rows();++n){
        double sign=n%2?-1:1;value+=rho(n,n).real()*sign*laguerre(n,0,2*rr);
        C power=1;double ratio=1;
        for(int m=n+1;m<rho.rows();++m){power*=z;ratio/=std::sqrt(double(m));
            value+=2*std::real(rho(m,n)*power)*ratio*sign*laguerre(n,m-n,2*rr);}
    }
    return std::exp(-rr)*value/(pi*hbar);
}
}
