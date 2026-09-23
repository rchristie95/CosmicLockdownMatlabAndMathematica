#include "fock.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
using namespace fock;
namespace fs=std::filesystem;
std::vector<double> numbers(const std::string& s){
    std::string text=s;std::replace(text.begin(),text.end(),',',' ');std::istringstream in(text);
    std::vector<double> v;double x;while(in>>x){check(std::isfinite(x),"Nonfinite input");v.push_back(x);}check(in.eof(),"Invalid numeric list");return v;
}
std::vector<double> read_numbers(const std::string& file){std::ifstream f(file);check(bool(f),"Cannot open "+file);std::ostringstream s;s<<f.rdbuf();return numbers(s.str());}
void matrix_write(std::ofstream& f,const M& m){
    // Explicit real/imaginary encoding, column-major; independent of complex ABI.
    for(Eigen::Index j=0;j<m.cols();++j)for(Eigen::Index i=0;i<m.rows();++i){double a=m(i,j).real(),b=m(i,j).imag();f.write(reinterpret_cast<char*>(&a),8);f.write(reinterpret_cast<char*>(&b),8);}
}
void save(const Config& c,const Result& r,const fs::path& out){
    fs::create_directories(out);Ops op(c.basis,c);Eigen::SelfAdjointEigenSolver<M> eigx(op.x);
    {std::ofstream f(out/"operators.f64",std::ios::binary);
     for(const M& a:std::vector<M>{op.x,op.p,op.H(c.initial,c),op.coupling})matrix_write(f,a);
     check(bool(f),"Cannot write operators");}
    M projector=M::Zero(c.basis,c.basis);for(int j=0;j<c.basis;++j)if(eigx.eigenvalues()(j)>1e-12)projector+=density(V(eigx.eigenvectors().col(j)));
    std::ofstream states(out/"psi.f64",std::ios::binary),rhos(out/"rho.f64",std::ios::binary),obs(out/"observables.csv");
    obs<<std::setprecision(17)<<"N,trace,purity,mean_phi,mean_momentum,variance_phi,Pfalse,energy,min_eigenvalue\n";
    for(const auto& frame:r.frames){
        matrix_write(rhos,frame.rho);if(frame.psi.size())matrix_write(states,frame.psi);
        const M& rho=frame.rho;double tr=rho.trace().real();M rn=rho/tr;
        double mean=(op.x*rn).trace().real();Eigen::SelfAdjointEigenSolver<M> eigen((rho+rho.adjoint())*.5);
        obs<<frame.t<<','<<tr<<','<<(rn*rn).trace().real()<<','<<mean<<','<<(op.p*rn).trace().real()<<','
           <<(op.x*op.x*rn).trace().real()-mean*mean<<','<<(projector*rn).trace().real()<<','
           <<(op.H(frame.t,c)*rn).trace().real()<<','<<eigen.eigenvalues()(0)<<'\n';
    }
    if(!r.noise.empty()){std::ofstream f(out/"noise.txt");f<<std::setprecision(17);for(double v:r.noise)f<<v<<'\n';}
    {std::ofstream f(out/"bath_noise.txt");f<<std::setprecision(17);for(C v:r.zeta)f<<v.real()<<' '<<v.imag()<<'\n';}
    std::ofstream info(out/"metadata.json");info<<std::setprecision(17);
    info<<"{\n\"format\":1,\"solver\":\"Fock basis\",\"workflow\":\""<<c.workflow<<"\",\"model\":\""<<c.model<<"\",\n"
        <<"\"basis\":"<<c.basis<<",\"frames\":"<<r.frames.size()<<",\"has_psi\":"<<(r.frames[0].psi.size()?"true":"false")<<",\n"
        <<"\"hbar\":"<<c.hbar<<",\"mu\":"<<c.mu<<",\"beta3\":"<<c.beta3<<",\"beta4\":"<<c.beta4<<",\"lambda\":"<<c.lambda<<",\"H\":"<<c.H<<",\"volume\":"<<c.vol()<<",\"omega\":"<<c.omega<<",\n"
        <<"\"initial\":"<<c.initial<<",\"final\":"<<c.final<<",\"step\":"<<c.step<<",\"seed\":"<<c.seed<<",\"grid\":"<<c.grid<<",\"extent\":"<<c.extent<<",\n"
        <<"\"rtol\":"<<c.rtol<<",\"atol\":"<<c.atol<<",\"accepted_steps\":"<<r.stats.accepted<<",\"rejected_steps\":"<<r.stats.rejected<<",\n"
        <<"\"integrator\":\""<<(c.workflow=="sse"?"Gaussian instrument / spectral Strang":c.workflow=="lindblad"?"CPTP spectral Strang":c.nm()?"rank-factorized Dormand-Prince 5(4)":c.workflow=="closed"?"fourth-order spectral composition":"Hermitian eigensolver")<<"\",\n"
        <<"\"ground_relative_residual\":"<<r.residual<<",\"wigner_exported\":"<<(c.wigner?"true":"false")<<",\n"
        <<"\"noise_convention\":\"one standard normal quantile per Gaussian Born-mixture measurement step; q=integral Gamma/hbar dN\",\n"
        <<"\"memory_initialization\":\"zero at model activation; adaptive continuous memory integral\",\n"
        <<"\"encoding\":\"native-endian float64 (supported CI hosts little-endian); complex real/imag pairs; Fock matrices column-major within each frame\",\n"
        <<"\"monitoring_start\":"<<((c.nm()&&c.workflow=="nm-sse")||(!c.nm()&&c.power()!=1)?std::max(-1.,c.initial):c.initial)<<",\n"
        <<"\"N_crossover\":"<<std::log(1/(2*c.vol()*c.vol()*std::pow(c.mu,6)))/6<<",\n"
        <<"\"projection\":\"closed outputs retain projected norm; other outputs normalized; dynamics basis 2b for SSE/adiabatic/NM preparation and 3b for closed/GKLS preparation\",\n\"interval_steps\":[";
    for(size_t j=0;j<c.steps.size();++j){if(j)info<<',';info<<c.steps[j];}info<<"]\n}\n";
    if(c.wigner){
        std::ofstream w(out/"wigner.f64",std::ios::binary),axis(out/"phase_axis.f64",std::ios::binary);
        for(int j=0;j<c.grid;++j){double x=-c.extent+2*c.extent*j/(c.grid-1);axis.write(reinterpret_cast<char*>(&x),8);}
        for(const auto& frame:r.frames)for(int k=0;k<c.grid;++k)for(int j=0;j<c.grid;++j){
            double x=-c.extent+2*c.extent*j/(c.grid-1),p=-c.extent+2*c.extent*k/(c.grid-1);
            double a=wigner(frame.rho,x,p,c.hbar);w.write(reinterpret_cast<char*>(&a),8);
        }
        check(bool(w)&&bool(axis),"Cannot write Wigner output");
    }
    check(bool(states)&&bool(rhos)&&bool(obs)&&bool(info),"Cannot write output");
    std::cout<<c.workflow<<" / "<<c.model<<": "<<r.frames.size()<<" frames -> "<<out.string()<<'\n';
}
int selftest(){
    Config c;c.basis=6;c.initial=-.2;c.final=-.19;c.step=.001;c.frames=3;c.wigner=false;
    Ops o(6,c);check((o.x-o.x.adjoint()).norm()<1e-14,"X Hermiticity");
    V p=ground(o.H(0,c));check((o.H(0,c)*p-p*(p.dot(o.H(0,c)*p))).norm()<1e-12,"Eigen residual");
    Spectral spectral(o,c);V q=spectral.closed(p,.2,.01,c);check(std::abs(q.norm()-1)<1e-14,"Unitary norm");
    M r=spectral.channel(density(p),.2,.1,c,.004);Eigen::SelfAdjointEigenSolver<M> e(r);
    check(std::abs(r.trace()-C(1))<1e-12&&e.eigenvalues()(0)>-1e-12,"GKLS trace/positivity");
    M gaussian=M::Zero(3,3);gaussian(0,0)=1;
    check(std::abs(wigner(gaussian,.7,.2,1)-std::exp(-.53)/pi)<1e-14,"Gaussian Wigner");
    M large=M::Zero(400,400);large(0,0)=1;
    check(std::abs(wigner(large,10,10,1)-std::exp(-200.)/pi)<1e-95,"Large-basis Wigner overflow");
    large.setZero();large(399,399)=1;
    check(std::abs(wigner(large,0,0,1)+1/pi)<1e-12&&std::isfinite(wigner(large,28,0,1)),"High Fock Wigner normalization/scaling");
    V coherent(3);coherent<<1,I,0;coherent.normalize();double moment=0,integral=0,dx=.08;
    for(int i=-75;i<=75;++i)for(int j=-75;j<=75;++j){double w=wigner(density(coherent),i*dx,j*dx,1);integral+=w*dx*dx;moment+=j*dx*w*dx*dx;}
    check(std::abs(integral-1)<1e-10&&std::abs(moment-1/std::sqrt(2.))<1e-10,"Wigner sign and momentum marginal");
    for(std::string model:{"x","x2","x3"})for(std::string wf:{"adiabatic","closed","sse","lindblad"}){c.model=model;c.workflow=wf;auto result=solve(c);check(result.frames.back().rho.allFinite(),"Workflow finite");}
    c.model="x";c.workflow="nm-density";auto a=solve(c);c.rtol=1e-11;c.atol=1e-13;auto b=solve(c);
    check((a.frames.back().rho-b.frames.back().rho).norm()<1e-8,"Memory tolerance refinement");
    c.workflow="nm-sse";solve(c);
    // Nonzero-coupling ensemble against an independent exact two-state channel.
    M l=M::Zero(2,2),h=M::Zero(2,2);l(0,0)=-1;l(1,1)=1;V v(2);v<<std::sqrt(.4),std::sqrt(.6);
    std::mt19937_64 rng(91);std::normal_distribution<double> normal;M avg=M::Zero(2,2);double dt=1e-4;
    for(int k=0;k<40000;++k){V z=measurement(v,l.diagonal().real(),dt,normal(rng));avg+=density(z)/40000.;}
    M exact=density(v);exact(0,1)*=std::exp(-2*dt);exact(1,0)=exact(0,1);
    check((avg-exact).norm()<.0005,"SSE ensemble dephasing");
    for(int power=1;power<=3;++power){
        M lp=M::Zero(2,2);lp(0,0)=std::pow(-1.,power);lp(1,1)=std::pow(2.,power);
        M target=density(v);target(0,1)*=std::exp(-.5*std::norm(lp(0,0)-lp(1,1))*.002);target(1,0)=target(0,1);
        for(int steps:{20,40}){
            double sum=0,sum2=0,pops=0,pops2=0;const int samples=6000;
            double step=.002/steps;
            for(int trial=0;trial<samples;++trial){
                V state=v;
                for(int j=0;j<steps;++j)state=measurement(state,lp.diagonal().real(),step,normal(rng));
                double coherence=(state(0)*std::conj(state(1))).real(),population=std::norm(state(0));
                sum+=coherence;sum2+=coherence*coherence;pops+=population;pops2+=population*population;
            }
            double se=std::sqrt(std::max(0.,sum2/samples-std::pow(sum/samples,2))/samples);
            double pse=std::sqrt(std::max(0.,pops2/samples-std::pow(pops/samples,2))/samples);
            double error=std::abs(sum/samples-target(0,1).real());
            check(error<4*se+3e-4&&std::abs(pops/samples-.4)<4*pse+3e-4,"Refined ensemble outside Monte Carlo uncertainty and step tolerance");
            std::cout<<"Ensemble X^"<<power<<" dt="<<step<<" coherence_error="<<error<<" MC_SE="<<se<<"\n";
        }
    }
    // Strong measurement: exact Born populations and decoherence, even when
    // an explicit drift step is grossly outside its stability range.
    for(double clock:{.01,.1,1.,10.}){
        R eigenvalues(2);eigenvalues<<-1,8;double sum=0,sum2=0,coh=0,coh2=0;int count=16000;
        for(int j=0;j<count;++j){V state=measurement(v,eigenvalues,clock,normal(rng));double pop=std::norm(state(0)),off=(state(0)*std::conj(state(1))).real();sum+=pop;sum2+=pop*pop;coh+=off;coh2+=off*off;}
        double se=std::sqrt(std::max(0.,sum2/count-std::pow(sum/count,2))/count);
        double ce=std::sqrt(std::max(0.,coh2/count-std::pow(coh/count,2))/count);
        check(std::abs(sum/count-.4)<5*se+1e-10,"Strong measurement Born population");
        check(std::abs(coh/count-std::sqrt(.24)*std::exp(-40.5*clock))<5*ce+1e-10,"Strong measurement dephasing");
    }
    std::cout<<"PASS: Fock operators, ground, exponential, GKLS, signed Wigner, all models, memory and SSE ensemble\n";return 0;
}
int main(int argc,char**argv){try{
    Config c;fs::path output="fock_output";std::string sweep="lambda",values;
    for(int k=1;k<argc;++k){std::string arg=argv[k];
        if(arg=="--self-test")return selftest();
        if(arg=="--help"){std::cout<<"cosmic_fock --workflow adiabatic|closed|sse|lindblad|sweep|nm-sse|nm-density --model x|x2|x3\n"
            <<"--rtol V --atol V (NM error tolerances) --basis N --initial N --final N --step dN --frames N --times CSV --steps CSV --seed N --noise FILE --bath-noise FILE\n"
            <<"--hbar V --mu V --beta3 V --beta4 V --lambda V --hubble V --volume V --omega V --grid N --extent V --no-wigner --output DIR\n"
            <<"--sweep lambda|hubble --values CSV; Hubble zero is the adiabatic reference.\n";return 0;}
        if(arg=="--no-wigner"){c.wigner=false;continue;}
        check(k+1<argc,"Missing argument value");std::string v=argv[++k];
        if(arg=="--workflow")c.workflow=v;else if(arg=="--model")c.model=v;else if(arg=="--output")output=v;
        else if(arg=="--basis")c.basis=std::stoi(v);else if(arg=="--frames")c.frames=std::stoi(v);else if(arg=="--grid")c.grid=std::stoi(v);
        else if(arg=="--initial")c.initial=std::stod(v);else if(arg=="--final")c.final=std::stod(v);else if(arg=="--step")c.step=std::stod(v);
        else if(arg=="--hbar")c.hbar=std::stod(v);else if(arg=="--mu")c.mu=std::stod(v);else if(arg=="--beta3")c.beta3=std::stod(v);else if(arg=="--beta4")c.beta4=std::stod(v);
        else if(arg=="--lambda")c.lambda=std::stod(v);else if(arg=="--hubble")c.H=std::stod(v);else if(arg=="--volume")c.volume=std::stod(v);
        else if(arg=="--rtol")c.rtol=std::stod(v);else if(arg=="--atol")c.atol=std::stod(v);
        else if(arg=="--omega")c.omega=std::stod(v);else if(arg=="--extent")c.extent=std::stod(v);else if(arg=="--seed")c.seed=std::stoull(v);
        else if(arg=="--times")c.times=numbers(v);else if(arg=="--steps")c.steps=numbers(v);else if(arg=="--noise")c.noise=read_numbers(v);
        else if(arg=="--bath-noise"){auto a=read_numbers(v);check(a.size()==8,"Bath noise requires four real/imaginary pairs");for(int j=0;j<4;++j)c.zeta.emplace_back(a[2*j],a[2*j+1]);}
        else if(arg=="--sweep")sweep=v;else if(arg=="--values")values=v;else throw std::runtime_error("Unknown option "+arg);
    }
    check(c.grid>=3&&c.grid<=1025&&c.extent>0&&std::isfinite(c.extent),"Invalid phase-space grid");
    if(c.workflow=="sweep"){
        check(sweep=="lambda"||sweep=="hubble","Unknown sweep");auto vals=numbers(values.empty()?(sweep=="lambda"?"0,.05,.1,.15,.2":"0,.25,.5,5"):values);
        check(!vals.empty(),"Empty sweep");fs::create_directories(output);std::ofstream index(output/"sweep.csv");index<<"parameter,value,directory,workflow\n";
        for(size_t j=0;j<vals.size();++j){Config sub=c;sub.workflow="lindblad";
            if(sweep=="lambda")sub.lambda=vals[j];else if(vals[j]==0){sub.workflow="adiabatic";sub.H=c.H;}else sub.H=vals[j];
            auto dir="run-"+std::to_string(j);save(sub,solve(sub),output/dir);index<<sweep<<','<<vals[j]<<','<<dir<<','<<sub.workflow<<'\n';}
    }else save(c,solve(c),output);
    return 0;
}catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}}
