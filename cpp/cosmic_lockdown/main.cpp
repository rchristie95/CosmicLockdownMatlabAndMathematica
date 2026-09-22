// C++17 position-monitoring solver. Default: arXiv:2512.14204v2, Eqs. 2.7, 3.15.
// --model repository preserves the different generator at commit d382d97.
// Fourier splitting is an alternative to the paper's truncated Fock basis.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using C = std::complex<double>;
using CV = std::vector<C>;
using RV = std::vector<double>;
constexpr double PI = 3.1415926535897932384626433832795;
constexpr C I(0, 1);
constexpr double MU=.5, B3=.025, B4=.13;

struct Config {
    size_t n=32768, trajectories=1;
    double extent=48, step=.001, hubble=5, coupling=.05, final_n=1;
    // Selected press example; provenance lists the 16-trajectory candidate search.
    uint64_t seed=20260928;
    bool paper=true, export_wigner=true, near_minimum=true;
    double volume() const { return paper?4*std::sqrt(2.):1.; }
    double monitoring_start() const { return paper?-2.:-1.; }
    std::filesystem::path output="cpp_output";
};

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
double norm2(const CV& a) { double s=0; for (auto z:a) s+=std::norm(z); return s; }
void normalise(CV& a) {
    double n=std::sqrt(norm2(a)); require(std::isfinite(n)&&n>0,"Invalid state norm");
    for(auto& z:a) z/=n;
}
double dot(const RV& a,const RV& b) { double s=0; for(size_t j=0;j<a.size();++j)s+=a[j]*b[j]; return s; }

void fft(CV& a, bool inverse=false) {
    const size_t n=a.size(); require(n && !(n&(n-1)),"FFT size must be a power of two");
    for(size_t j=1,k=0;j<n;++j) {
        size_t bit=n>>1; for(;k&bit;bit>>=1) k^=bit; k^=bit;
        if(j<k)std::swap(a[j],a[k]);
    }
    for(size_t len=2;len<=n;len<<=1) {
        C step=std::polar(1.,(inverse?2:-2)*PI/double(len));
        for(size_t start=0;start<n;start+=len) {
            C w=1.;
            for(size_t k=0;k<len/2;++k) {
                C u=a[start+k],v=a[start+k+len/2]*w;
                a[start+k]=u+v; a[start+k+len/2]=u-v; w*=step;
            }
        }
    }
    if(inverse)for(auto& z:a)z/=double(n);
}

double potential(double x) { return -.5*MU*MU*x*x+2*B3*MU*x*x*x/3+(B4*B4-B3*B3)*std::pow(x,4)/4; }
double gamma(double N,double lambda) { return 131*PI*lambda*lambda*std::exp(6*N)/(512*std::pow(MU,5)); }
double gamma(double N,const Config& c) { return gamma(N,c.coupling)*(c.paper?2/c.volume():1.); }
double crossover(const Config& c) { return std::log(1/(2*c.volume()*c.volume()*std::pow(MU,6)))/6; }

struct Grid {
    Config c; RV x,p,v,kin;
    explicit Grid(const Config& config):c(config),x(c.n),p(c.n),v(c.n),kin(c.n) {
        for(size_t j=0;j<c.n;++j) {
            x[j]=-c.extent+2*c.extent*j/c.n;
            auto mode=j<c.n/2?static_cast<long>(j):static_cast<long>(j)-static_cast<long>(c.n);
            p[j]=PI*mode/c.extent; v[j]=potential(x[j]); kin[j]=p[j]*p[j]/(2*c.hubble*c.volume());
        }
    }
    RV initial_h(const RV& a,double shift=0) const {
        CV z(a.begin(),a.end()); fft(z);
        for(size_t j=0;j<c.n;++j)z[j]*=std::exp(6)*kin[j];
        fft(z,true);
        RV r(c.n); for(size_t j=0;j<c.n;++j)r[j]=z[j].real()+(std::exp(-6)*c.volume()*v[j]/c.hubble-shift)*a[j];
        return r;
    }
    RV precondition(const RV& a,double shift) const {
        CV z(a.begin(),a.end()); fft(z);
        for(size_t j=0;j<c.n;++j)z[j]/=(std::exp(6)*kin[j]+.1-shift);
        fft(z,true);
        RV r(c.n); for(size_t j=0;j<c.n;++j)r[j]=z[j].real(); return r;
    }
    RV inverse(const RV& rhs,double shift) const {
        RV solution(c.n,0),r=rhs,z=precondition(r,shift),d=z;
        double rz=dot(r,z),target=dot(rhs,rhs)*1e-25;
        for(int k=0;k<3000;++k) {
            RV ad=initial_h(d,shift); double alpha=rz/dot(d,ad);
            for(size_t j=0;j<c.n;++j){solution[j]+=alpha*d[j];r[j]-=alpha*ad[j];}
            if(dot(r,r)<target)return solution;
            z=precondition(r,shift); double next=dot(r,z),beta=next/rz;
            for(size_t j=0;j<c.n;++j)d[j]=z[j]+beta*d[j];
            rz=next;
        }
        throw std::runtime_error("Ground-state inverse solve did not converge");
    }
    CV ground(double& energy,double& residual) const {
        RV a(c.n);for(size_t j=0;j<c.n;++j)a[j]=std::exp(-x[j]*x[j]/225);
        double shift=std::exp(-6)*c.volume()*potential(-MU/(B4-B3))/c.hubble-.01;
        for(int k=0;k<100;++k) {
            a=inverse(a,shift); double length=std::sqrt(dot(a,a));for(auto& z:a)z/=length;
            RV h=initial_h(a);energy=dot(a,h);for(size_t j=0;j<c.n;++j)h[j]-=energy*a[j];
            residual=std::sqrt(dot(h,h));
            if(residual<std::max(1e-8,1e-7*double(c.n)/16384))return CV(a.begin(),a.end());
        }
        throw std::runtime_error("Ground-state residual failed tolerance");
    }
};

void unitary(CV& psi,const CV& phase,const CV& kinetic_phase) {
    for(size_t j=0;j<psi.size();++j)psi[j]*=phase[j];
    fft(psi);
    for(size_t j=0;j<psi.size();++j)psi[j]*=kinetic_phase[j];
    fft(psi,true);
    for(size_t j=0;j<psi.size();++j)psi[j]*=phase[j];
}

// A half-step of Gamma D[X]. q = integral(Gamma/2) over the FULL step.
// Measurement amplitude exp[-q*x^2 + 2*q*m*x], with m sampled from its exact Born law.
// Averaging over m multiplies rho(x,y) by exp[-q*(x-y)^2/2] for this half-step.
void measure(CV& psi,const RV& x,double q,std::mt19937_64& rng) {
    if(q==0)return;
    std::uniform_real_distribution<double> uniform(0,1);
    std::normal_distribution<double> gaussian(0,1);
    double target=uniform(rng)*norm2(psi),sum=0;size_t k=0;
    for(;k+1<psi.size();++k){sum+=std::norm(psi[k]);if(sum>=target)break;}
    double m=x[k]+gaussian(rng)/std::sqrt(4*q),largest=-1e300;
    for(auto y:x)largest=std::max(largest,-q*y*y+2*q*m*y);
    for(size_t j=0;j<psi.size();++j)psi[j]*=std::exp(-q*x[j]*x[j]+2*q*m*x[j]-largest);
    normalise(psi);
}

struct Moments { double right=0,mean=0,var=0,edge=0; };
Moments moments(const CV& psi,const Grid& g) {
    Moments m; double total=norm2(psi);
    for(size_t j=0;j<psi.size();++j) {
        double w=std::norm(psi[j])/total;
        m.right+=w*(g.x[j]>1e-10?1:(std::abs(g.x[j])<1e-10?.5:0));
        m.mean+=w*g.x[j];m.var+=w*g.x[j]*g.x[j];
        if(std::abs(g.x[j])>.85*g.c.extent)m.edge+=w;
    }
    m.var-=m.mean*m.mean;return m;
}

struct Wigner {
    RV data,p;
    size_t stride=1,columns=0;
    double integral=0,negative_volume=0,marginal_error=0,imaginary_error=0,minimum=1e300,maximum=-1e300;
};
Wigner wigner(const CV& psi,double dx) {
    const size_t n=psi.size(),m=2*n;
    CV f=psi;fft(f);CV up(m,0);
    for(size_t k=0;k<n/2;++k)up[k]=f[k];
    for(size_t k=n/2+1;k<n;++k)up[k+n]=f[k];
    up[n/2]=f[n/2]/2.;up[3*n/2]=f[n/2]/2.;
    fft(up,true);for(auto& z:up)z*=std::sqrt(2.);
    // Retain <=16384 spatial columns for export; marginal/integral tests catch undersampling.
    Wigner w;w.stride=std::max(size_t(1),n/16384);w.columns=n/w.stride;
    const double dp=PI/(n*dx);
    std::vector<long> output_row(m,-1);
    for(size_t r=0;r<m;++r) {
        double p=(static_cast<double>(r)-n)*dp;
        if(n<=256 || std::abs(p)<=32) {output_row[r]=static_cast<long>(w.p.size());w.p.push_back(p);}
    }
    w.data.resize(w.p.size()*w.columns);
    for(size_t j=0;j<n;j+=w.stride) {
        CV corr(m,0);
        for(long k=-static_cast<long>(n);k<static_cast<long>(n);++k) {
            long a=2*static_cast<long>(j)+k,b=2*static_cast<long>(j)-k;
            if(a>=0&&b>=0&&a<static_cast<long>(m)&&b<static_cast<long>(m))
                corr[static_cast<size_t>((k+static_cast<long>(m))%static_cast<long>(m))]=up[a]*std::conj(up[b]);
        }
        fft(corr);double marginal=0;
        for(size_t r=0;r<m;++r) {
            C value=corr[(r+n)%m]/PI;double v=value.real();
            if(output_row[r]>=0)w.data[static_cast<size_t>(output_row[r])*w.columns+j/w.stride]=v;
            w.minimum=std::min(w.minimum,v);w.maximum=std::max(w.maximum,v);
            w.imaginary_error=std::max(w.imaginary_error,std::abs(value.imag()));
            w.integral+=v*dx*w.stride*dp;marginal+=v*dp;
            if(v<0)w.negative_volume-=v*dx*w.stride*dp;
        }
        w.marginal_error=std::max(w.marginal_error,std::abs(marginal-std::norm(psi[j])/dx));
    }
    return w;
}
void binary(const std::filesystem::path& path,const RV& a) {
    std::ofstream f(path,std::ios::binary);require(bool(f),"Cannot write "+path.string());
    f.write(reinterpret_cast<const char*>(a.data()),std::streamsize(a.size()*sizeof(double)));
    require(bool(f),"Incomplete binary write");
}
void wigner_json(std::ostream& f,const Wigner& w) {
    f<<"{\"integral\":"<<w.integral<<",\"negative_volume\":"<<w.negative_volume
     <<",\"marginal_error\":"<<w.marginal_error<<",\"imaginary_error\":"<<w.imaginary_error
     <<",\"min\":"<<w.minimum<<",\"max\":"<<w.maximum<<"}";
}

int self_test() {
    CV a(256);for(size_t j=0;j<a.size();++j)a[j]=C(std::sin(.3*j),std::cos(.2*j));
    CV b=a;fft(b);fft(b,true);double error=0;for(size_t j=0;j<a.size();++j)error=std::max(error,std::abs(a[j]-b[j]));
    require(error<1e-12,"FFT roundtrip failed");
    const double dx=20./256;CV psi(256);
    for(size_t j=0;j<256;++j){double x=-10+j*dx;psi[j]=std::exp(-x*x/2)*std::sqrt(dx)/std::pow(PI,.25);}
    Wigner w=wigner(psi,dx);double analytical=0;
    for(size_t r=0;r<512;++r)for(size_t j=0;j<256;++j){double x=-10+j*dx;analytical=std::max(analytical,std::abs(w.data[r*256+j]-std::exp(-x*x-w.p[r]*w.p[r])/PI));}
    require(analytical<1e-12&&std::abs(w.integral-1)<1e-11,"Analytic Gaussian Wigner test failed");
    std::mt19937_64 rng(1); CV untouched=psi;measure(untouched,RV(256),0,rng);
    require(untouched==psi,"Zero coupling modified state");
    // The Fourier kinetic phase must preserve norm.
    CV ph(256),pk(256);for(size_t j=0;j<256;++j){ph[j]=std::polar(1.,.03*j);pk[j]=std::polar(1.,-.02*j*j);}
    unitary(untouched,ph,pk);require(std::abs(norm2(untouched)-norm2(psi))<1e-12,"Unitary norm test failed");
    // Nonzero-coupling test of the actual Born sampler, including its factor of two.
    // Two independent half-steps must damp rho01 by exp[-q*(x0-x1)^2].
    const RV positions={-1.,1.}; const double q=.08; const size_t trials=200000;
    double population=0,coherence=0;
    std::mt19937_64 channel_rng(712);
    for(size_t trial=0;trial<trials;++trial) {
        CV state={std::sqrt(.4),std::sqrt(.6)};
        measure(state,positions,q,channel_rng);measure(state,positions,q,channel_rng);
        population+=std::norm(state[0]);coherence+=(state[0]*std::conj(state[1])).real();
    }
    population/=trials;coherence/=trials;
    const double expected=std::sqrt(.24)*std::exp(-4*q);
    require(std::abs(population-.4)<.003 && std::abs(coherence-expected)<.003,"Nonzero measurement ensemble disagrees with exact dephasing channel");
    Config paper,repo;repo.paper=false;
    require(std::abs(gamma(1,paper)/gamma(1,repo)-2/(4*std::sqrt(2.)))<1e-14,"Paper/repository rate mapping failed");
    require(paper.monitoring_start()==-2 && repo.monitoring_start()==-1,"Monitoring schedules failed");
    require(std::abs(crossover(paper))<1e-14,"Paper crossover is not N=0");
    require(std::abs(crossover(repo)-std::log(32.)/6)<1e-14,"Repository crossover mismatch");
    // Algebraic mapping requires BOTH a time shift and a coupling change.
    const double shift=std::log(paper.volume())/3;
    Config mapped=repo;mapped.coupling=std::sqrt(2.)*paper.coupling/std::pow(paper.volume(),1.5);
    double mapping_error=0;
    for(double N:{-2.,0.,1.}) {
        double kp=std::exp(-3*N)/(2*paper.hubble*paper.volume());
        double kr=std::exp(-3*(N+shift))/(2*repo.hubble);
        double vp=std::exp(3*N)*paper.volume()/paper.hubble;
        double vr=std::exp(3*(N+shift))/repo.hubble;
        mapping_error=std::max({mapping_error,std::abs(kr/kp-1),std::abs(vr/vp-1),std::abs(gamma(N+shift,mapped)/gamma(N,paper)-1)});
    }
    require(mapping_error<1e-13,"Time-origin/volume/coupling mapping failed");
    std::cout<<"PASS: Nstar paper="<<crossover(paper)<<", repository="<<crossover(repo)<<", time/volume/coupling mapping relative error="<<mapping_error<<"\n";
    std::cout<<"PASS: nonzero measurement ensemble population="<<population<<", coherence="<<coherence<<", exact="<<expected<<"; paper/repository coefficients\n";
    std::cout<<"PASS: FFT roundtrip="<<error<<", Gaussian Wigner max error="<<analytical
             <<", Wigner integral="<<w.integral<<", lambda=0 identity, unitary norm\n";return 0;
}

int run(const Config& c) {
    require(c.n>=128 && !(c.n&(c.n-1)),"--grid must be a power of two >=128");
    require(c.trajectories>0&&c.trajectories<=1024,"Invalid trajectory count");
    require(c.step>0&&c.extent>0&&c.hubble>0&&c.coupling>=0&&c.final_n>-1,"Invalid physical/numerical parameter");
    std::filesystem::create_directories(c.output);
    auto start=std::chrono::steady_clock::now();Grid g(c);double energy=0,residual=0;
    CV off=g.ground(energy,residual);std::vector<CV> on(c.trajectories,off);
    std::vector<std::mt19937_64> rng;for(size_t r=0;r<c.trajectories;++r)rng.emplace_back(c.seed+r);
    std::ofstream history(c.output/"history.csv");history<<std::setprecision(17)<<"N,Gamma_on,Pfalse_off";
    for(size_t r=0;r<c.trajectories;++r)history<<",Pfalse_trajectory_"<<r;
    history<<"\n";
    double max_edge=moments(off,g).edge,max_norm_error=0,integrated_gamma=0;
    std::cout<<"Ground E="<<energy<<", residual="<<residual<<"; grid="<<c.n<<std::endl;
    const size_t segments=2;
    for(size_t segment=0;segment<segments;++segment) {
        double begin=segment==0?-2:-1,end=segment==0?-1:c.final_n;
        size_t steps=static_cast<size_t>(std::ceil((end-begin)/c.step));double dt=(end-begin)/steps;
        for(size_t k=0;k<steps;++k) {
            double a=begin+k*dt,b=a+dt;
            double iv=c.volume()*std::exp(3*a)*std::expm1(3*dt)/(3*c.hubble);
            double ik=std::exp(-3*a)*(-std::expm1(-3*dt))/3;
            double q=a<c.monitoring_start()?0:gamma(a,c)*std::expm1(6*dt)/12;
            integrated_gamma+=2*q;
            CV phase(c.n),pk(c.n);
            for(size_t j=0;j<c.n;++j){phase[j]=std::polar(1.,-.5*iv*g.v[j]);pk[j]=std::polar(1.,-ik*g.kin[j]);}
            unitary(off,phase,pk);
            for(size_t r=0;r<c.trajectories;++r){measure(on[r],g.x,q,rng[r]);unitary(on[r],phase,pk);measure(on[r],g.x,q,rng[r]);}
            if(k%std::max(size_t(1),steps/20)==0||k+1==steps) {
                auto m=moments(off,g);max_edge=std::max(max_edge,m.edge);max_norm_error=std::max(max_norm_error,std::abs(norm2(off)-1));
                history<<b<<","<<(a<c.monitoring_start()?0:gamma(b,c))<<","<<m.right;
                for(const auto& psi:on){auto t=moments(psi,g);history<<","<<t.right;max_edge=std::max(max_edge,t.edge);max_norm_error=std::max(max_norm_error,std::abs(norm2(psi)-1));}history<<"\n";
            }
        }
        std::cout<<"Completed N="<<end<<"; Gamma_on="<<(end<=c.monitoring_start()?0:gamma(end,c))<<std::endl;
    }
    size_t selected=0;bool found=false;double best=1e300;
    std::ofstream outcomes(c.output/"trajectory_outcomes.csv");
    outcomes<<std::setprecision(17)<<"index,seed,Pfalse,mean_phi,variance_phi,mean_momentum\n";
    for(size_t r=0;r<on.size();++r){
        auto m=moments(on[r],g);CV f=on[r];fft(f);double mean_p=0;
        for(size_t j=0;j<c.n;++j)mean_p+=g.p[j]*std::norm(f[j])/c.n;
        outcomes<<r<<","<<c.seed+r<<","<<m.right<<","<<m.mean<<","<<m.var<<","<<mean_p<<"\n";
        double distance=std::abs(m.mean-MU/(B4+B3));
        bool eligible=m.right>.999 && (!c.near_minimum || (distance<1 && m.var<.5 && std::abs(mean_p)<10));
        if(eligible && (!found || (c.near_minimum && distance<best))){best=distance;selected=r;found=true;}
    }
    if(c.coupling>0)require(found,"No trajectory meets the documented selection rule; increase --trajectories or change --seed");
    CV chosen=on[selected];double zero_difference=0;
    if(c.coupling==0){for(size_t j=0;j<c.n;++j)zero_difference=std::max(zero_difference,std::abs(off[j]-chosen[j]));require(zero_difference<1e-11,"lambda=0 trajectories differ from closed system");}
    std::ofstream states(c.output/"states.csv");states<<std::setprecision(17)<<"phi,off_real,off_imag,on_real,on_imag\n";
    for(size_t j=0;j<c.n;++j)states<<g.x[j]<<","<<off[j].real()<<","<<off[j].imag()<<","<<chosen[j].real()<<","<<chosen[j].imag()<<"\n";
    states.flush();outcomes.flush();history.flush();
    require(bool(states)&&bool(outcomes)&&bool(history),"Could not write state/trajectory output");
    double dx=2*c.extent/c.n;Wigner wo,wn;
    if(c.export_wigner) {
    wo=wigner(off,dx);wn=wigner(chosen,dx);
    require(std::abs(wo.integral-1)<1e-8&&std::abs(wn.integral-1)<1e-8,"Wigner normalisation failed");
    require(wo.marginal_error<1e-8&&wn.marginal_error<1e-8,"Wigner marginal failed");
    RV displayed_x,displayed_v;
    for(size_t j=0;j<c.n;j+=wo.stride){displayed_x.push_back(g.x[j]);displayed_v.push_back(g.v[j]);}
    binary(c.output/"phi.f64",displayed_x);binary(c.output/"momentum.f64",wo.p);binary(c.output/"potential.f64",displayed_v);
    binary(c.output/"wigner_off.f64",wo.data);binary(c.output/"wigner_on.f64",wn.data);
    }
    auto mo=moments(off,g),mn=moments(chosen,g);
    double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::ofstream info(c.output/"metadata.json");info<<std::setprecision(17);
    info<<"{\n\"source_commit\":\"d382d97dca93768d08187b90ca30d97747bbc862\",\n"
        <<"\"solver\":\"C++17 Fourier split, exact Born Gaussian measurement\",\n"
        <<"\"model\":\""<<(c.paper?"paper":"repository")<<"\",\"volume\":"<<c.volume()<<",\"wigner_exported\":"<<(c.export_wigner?"true":"false")<<",\n"
        <<"\"N_crossover\":"<<crossover(c)<<",\"N_convention\":\""<<(c.paper?"paper crossover at zero; no additional shift":"repository original N; crossover not zero")<<"\",\n"
        <<"\"H\":"<<c.hubble<<",\"mu\":"<<MU<<",\"beta3\":"<<B3<<",\"beta4\":"<<B4<<",\"lambda_on\":"<<c.coupling<<",\"lambda_off\":0,\n"
        <<"\"N_initial\":-2,\"N_final\":"<<c.final_n<<",\"monitoring_start\":"<<c.monitoring_start()<<",\"state_grid\":"<<c.n<<",\"grid\":"<<wo.columns<<",\"momentum_grid\":"<<wo.p.size()<<",\"wigner_stride\":"<<wo.stride<<",\"extent\":"<<c.extent<<",\"max_step\":"<<c.step<<",\n"
        <<"\"seed\":"<<c.seed<<",\"trajectories\":"<<c.trajectories<<",\"selected_index\":"<<selected<<",\"selection\":\""<<(c.near_minimum?"nearest false minimum among Pfalse>0.999, abs(mean_phi-phiF)<1, variance<0.5, abs(mean_p)<10":"first Pfalse>0.999")<<"\",\n"
        <<"\"Gamma_final\":"<<gamma(c.final_n,c)<<",\"integrated_Gamma\":"<<integrated_gamma<<",\"ground_energy\":"<<energy<<",\"ground_residual\":"<<residual<<",\n"
        <<"\"Pfalse_off\":"<<mo.right<<",\"Pfalse_on_selected\":"<<mn.right<<",\"mean_phi_off\":"<<mo.mean<<",\"mean_phi_on\":"<<mn.mean<<",\"variance_phi_off\":"<<mo.var<<",\"variance_phi_on\":"<<mn.var<<",\n"
        <<"\"max_position_edge\":"<<max_edge<<",\"max_norm_error\":"<<max_norm_error<<",\"lambda_zero_path_difference\":";
    if(c.coupling==0)info<<zero_difference;else info<<"null";
    info<<",\"seconds\":"<<elapsed<<",\n\"wigner_off\":";
    if(c.export_wigner)wigner_json(info,wo);else info<<"null";
    info<<",\n\"wigner_on\":";
    if(c.export_wigner)wigner_json(info,wn);else info<<"null";
    info<<"\n}\n";
    std::cout<<std::setprecision(9)<<"Selected trajectory "<<selected<<": Pfalse="<<mn.right<<", Var(phi)="<<mn.var
             <<"; off Var(phi)="<<mo.var<<"; Wigner negativity off/on="<<wo.negative_volume<<" / "<<wn.negative_volume
             <<"; elapsed="<<elapsed<<" s\n";
    return 0;
}

int main(int argc,char** argv) {
    try {
        Config c;
        for(int k=1;k<argc;++k) {
            std::string arg=argv[k];
            if(arg=="--self-test")return self_test();
            if(arg=="--help") {std::cout<<"cosmic_lockdown [--model paper|repository] [--selection near-minimum|first-false] [--output DIR] [--grid 32768] [--step .001] [--trajectories 1] [--seed 20260928] [--lambda .05] [--hubble 5] [--final 1] [--extent 48] [--no-wigner] [--self-test]\n";return 0;}
            if(arg=="--no-wigner"){c.export_wigner=false;continue;}
            require(k+1<argc,"Missing value for "+arg);std::string value=argv[++k];
            if(arg=="--model"){require(value=="paper"||value=="repository","Model must be paper or repository");c.paper=value=="paper";}
            else if(arg=="--selection"){require(value=="near-minimum"||value=="first-false","Unknown selection rule");c.near_minimum=value=="near-minimum";}
            else if(arg=="--output")c.output=value;
            else if(arg=="--grid")c.n=std::stoull(value);
            else if(arg=="--step")c.step=std::stod(value);
            else if(arg=="--trajectories")c.trajectories=std::stoull(value);
            else if(arg=="--seed")c.seed=std::stoull(value);
            else if(arg=="--lambda")c.coupling=std::stod(value);
            else if(arg=="--hubble")c.hubble=std::stod(value);
            else if(arg=="--final")c.final_n=std::stod(value);
            else if(arg=="--extent")c.extent=std::stod(value);
            else throw std::runtime_error("Unknown argument: "+arg);
        }
        return run(c);
    } catch(const std::exception& e) {std::cerr<<"ERROR: "<<e.what()<<"\n";return 1;}
}
