function Psi_new = Ham_Step_Big(Psi, dN, H_big, hbarV)
    % Piecewise-constant Hamiltonian over [N, N+dN]:
    %   Ψ_{n+1} = exp(-i H(N) dN / ħ) Ψ_n
    Psi_new = expmv((-1i/hbarV) * H_big, Psi, dN);
    Psi_new = Psi_new / max(norm(Psi_new), eps);
end




%======================================================================
%  NMQSD_MasterEquation_pagemat.m
%----------------------------------------------------------------------
%  Vectorised master-equation integrator using page-tensor algebra.
%  Requires MATLAB R2020b+ for implicit-expansion / pagemtimes.
%======================================================================
