function W = RhoWigner(Rho,x,p,hbar)

%% Readme

% takes density matrix pagematrix as input along with row vectors defining x, and p grids and hbar value.
% suggested grid choice below 
% Nx=Number of points 
% dX=sqrt(2*pi*hbar/Nx);
% dP=2*pi*hbar/dX/Nx;
% 
% for n=1:Nx
%     x(n)=(-Nx/2+(n-1))*dX;
%     p(n)=(-Nx/2+(n-1))*dP; 
% end

%% Initialisation

N=size(Rho,3);
Nx = size(Rho,1); %   Get length of vector
x=x.';
p=p.';

%% conversion matrix from Fock basis

% for n=0:bSize-1
%     CMatrixFock2Pos(:,n+1)=sqrt(dX/2^(n)/factorial(n))*(pi*hbar)^(-1/4).*exp(-x.^2/2/hbar).*hermiteH(n,x/sqrt(hbar));
% end

%% Conversion
														
x=ifftshift(x);
W=zeros(Nx,Nx,N);
WaitMessage = parfor_wait(N, 'Waitbar', true); %initialise progress bar

for n=1:N % requires parallel  computing toolbox much faster
% for n=1:N
    [V,D]=eig(Rho(:,:,n)); %eigendecomposition of density matrix
    for m=1:Nx
    EX1 = ifft( (fft(V(:,m))*ones(1,Nx)).*exp( 1i*x*p.'/2/hbar ));					%   +ve shift
    EX2 = ifft( (fft(V(:,m))*ones(1,Nx)).*exp( -1i*x*p.'/2/hbar ));					%   -ve shift
    W(:,:,n) = W(:,:,n)+real(D(m,m)*(1/2/pi/hbar)*fftshift(fft(fftshift(EX1.*conj(EX2), 2), [], 2), 2))';		%   Wigner function
    end
    WaitMessage.Send % progress bar

end
WaitMessage.Destroy %close progress bar

end