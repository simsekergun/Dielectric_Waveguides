function [Ex, Ey, Ez, Hx, Hy, Hz, betas, neffs, tt] = Solver_Waveguide_PolyeigDS(x, y, dx, dy, eps, lambda, desired_modes, neff_min, neff_max, plot_fields, save_plot, filename)
Nx = length(x);
Ny = length(y);
Nxy = Nx * Ny;

k0 = 2 * pi / lambda;
beta_trial = 0.9 * k0 * neff_max;  % Initial guess for eigs

f = 3e8/lambda;
w = 2*pi*f;
mu0 = 4e-7*pi;
h_factor = 1i/w/mu0;

%%% Building of the operators
AA=ones(1,Nx*Ny);
BB=ones(1,Nx*Ny-1);
% BB(Ny:Ny:end)=0;
Axy=ones(1,(Nx-1)*Ny);

% % First derivative (fourth-order accurate)
DX1 = (-1/12) * spdiags(Axy, -2*Ny, Nxy, Nxy) + (8/12) * spdiags(Axy, -Ny, Nxy, Nxy) ...
    - (8/12) * spdiags(Axy, Ny, Nxy, Nxy) + (1/12) * spdiags(Axy, 2*Ny, Nxy, Nxy);

DY1 = (-1/12) * spdiags(BB, -2, Nxy, Nxy) + (8/12) * spdiags(BB, -1, Nxy, Nxy) ...
    - (8/12) * spdiags(BB, 1, Nxy, Nxy) + (1/12) * spdiags(BB, 2, Nxy, Nxy);

% Second derivative (fourth-order accurate)
DX2 = (-1/12) * spdiags(Axy, -2*Ny, Nxy, Nxy) + (16/12) * spdiags(Axy, -Ny, Nxy, Nxy) ...
    + (-30/12) * spdiags(AA, 0, Nxy, Nxy) + (16/12) * spdiags(Axy, Ny, Nxy, Nxy) ...
    + (-1/12) * spdiags(Axy, 2*Ny, Nxy, Nxy);

DY2 = (-1/12) * spdiags(BB, -2, Nxy, Nxy) + (16/12) * spdiags(BB, -1, Nxy, Nxy) ...
    + (-30/12) * spdiags(AA, 0, Nxy, Nxy) + (16/12) * spdiags(BB, 1, Nxy, Nxy) ...
    + (-1/12) * spdiags(BB, 2, Nxy, Nxy);


DX1 = DX1/dx;
DY1 = DY1/dy;
DX2 = DX2/dx/dx;
DY2 = DY2/dy/dy;

deps_dx_inveps= [zeros(Ny,2) -eps(:,1:end-4)+8*eps(:,2:end-3)-8*eps(:,4:end-1) + eps(:,5:end) zeros(Ny,2) ]./eps./(12*dx);
deps_dy_inveps = [zeros(2,Nx); -eps(1:end-4, :)+8*eps(2:end-3, :)-8*eps(4:end-1, :) + eps(5:end, :); zeros(2,Nx) ]./eps ./ (12*dy);
depsdx_inveps = spdiags(deps_dx_inveps(:), 0, Nxy, Nxy);
depsdy_inveps = spdiags(deps_dy_inveps(:), 0, Nxy, Nxy);


%% Building of the Hamiltonian
% Construct M, T, and I (sparse matrices)
eps_diag = spdiags(eps(:), 0, Nxy, Nxy);
H9 = DX2 + DY2 + eps_diag * k0^2;
H1 = H9 + DX1 * depsdx_inveps;
H2 = DX1 * depsdy_inveps;
H3 = sparse(Nxy, Nxy);
H4 = DY1 * depsdx_inveps;
H5 = H9 + DY1 * depsdy_inveps;
H7 = -1i * depsdx_inveps;
H8 = -1i * depsdy_inveps;

M = [H1, H2, H3; H4, H5, H3; H3, H3, H9];
T = [H3, H3, H3; H3, H3, H3; H7, H8, H3];
I = speye(3 * Nxy);

% Construct the augmented matrices for linear GEP
A = [M, T; sparse(3 * Nxy, 3 * Nxy), speye(3 * Nxy)];
B = [sparse(3 * Nxy, 3 * Nxy), speye(3 * Nxy); speye(3 * Nxy), sparse(3 * Nxy, 3 * Nxy)];


beta_guess = k0*neff_max-1e-8i;
nmodes = max([length(desired_modes), max(desired_modes)]);
% Solve the linear GEP using eigs (for sparse matrices)
opts.tol = 1e-12;
opts.maxit = 1000;
[V, D] = eigs(A, B, 2*nmodes,beta_guess,opts);


% % Solve the linear GEP using eigs (for sparse matrices)
% opts.tol = 1e-12;
% opts.maxit = 1000;
% % [V, D] = eigs(A, B, desired_modes, beta_trial, opts);
% % [V, D] = eigs(A, B, nmodes,'LR',opts);
% [V, D] = eigs(A, B, 2*nmodes,beta_guess,opts);

% Extract eigenvalues (β) and eigenvectors (E)
betas = diag(D);
psis = V(1:3 * Nxy, :);  % First 3*Nxy rows correspond to E

% Filter solutions within the desired neff range
neffs = real(betas) / k0;
valid_modes = (neffs >= neff_min) & (neffs <= neff_max);
if length(valid_modes)<1
    disp('no solution')
end
betas = betas(valid_modes);
psis = psis(:, valid_modes);
neffs = neffs(valid_modes);
nfound = length(neffs);

% Sort by descending neff
[neffs, idx] = sort(neffs, 'descend');
betas = betas(idx);
psis = psis(:, idx);
disp(neffs)

% Extract Ex, Ey, Ez for each mode
Ex = reshape(psis(1:Nxy, :), Ny, Nx, []);
Ey = reshape(psis(Nxy+1:2*Nxy, :), Ny, Nx, []);
Ez = reshape(psis(2*Nxy+1:3*Nxy, :), Ny, Nx, []);
Hx = [];
Hy = [];
Hz = [];

tt = [];

for i5 = 1:min([length(desired_modes), nfound])
    mm = desired_modes(i5);
    beta = betas(mm);
    neff = neffs(mm);

    Ex0=psis(1:Nxy,mm);
    Ey0=psis(Nxy+1:2*Nxy,mm);
    Ez0=psis(2*Nxy+1:3*Nxy, mm);

    Hx = h_factor* reshape( DY1*Ez0+1i*beta*Ey0,[Ny,Nx]);
    Hy = h_factor* reshape(-DX1*Ez0-1i*beta*Ex0,[Ny,Nx]);
    Hz = h_factor* reshape( DX1*Ey0-DY1*Ex0,[Ny,Nx]);

    Ex = reshape(Ex0,[Ny,Nx]);
    Ey = reshape(Ey0,[Ny,Nx]);
    Ez = reshape(Ez0,[Ny,Nx]);

    Power_z_sqrt = sqrt(real(sum(sum(  (Ex(:).*conj(Hy(:))-Ey(:).*conj(Hx(:)))*dx*dy  )))/lambda^2);

    Ex = Ex/Power_z_sqrt;
    Ey = Ey/Power_z_sqrt;
    Ez = Ez/Power_z_sqrt;
    Hx = Hx/Power_z_sqrt;
    Hy = Hy/Power_z_sqrt;
    Hz = Hz/Power_z_sqrt;

    % maxE = max([max(max(abs(Ex))) max(max(abs(Ey))) max(max(abs(Ez)))]);
    % sumEs = [sum(sum(abs(Ex).^2)) sum(sum(abs(Ey).^2)) sum(sum(abs(Ez).^2))];
    % sumHs = [sum(sum(abs(Hx).^2)) sum(sum(abs(Hy).^2)) sum(sum(abs(Hz).^2))];
    % Efractions = sumEs./sum(sumEs);
    % Hfractions = sumHs./sum(sumHs);
    % disp(['E fractions: ' num2str(Efractions)]);
    % disp(['H fractions: ' num2str(Hfractions)]);

    if plot_fields== 1
        textx = 1e6*(min(x)+ (max(x)-min(x))/15);
        texty = 1e6*(max(y) - (max(y)-min(y))/10);
        textx2 = 1e6*(min(x)+ (max(x)-min(x))/3);
        texty2 = 1e6*(min(y) + (max(y)-min(y))/10);

        figure('position',[400, 600, 1200, 500],'Name','WG_Solver');
        subplot(231);
        pcolor(x*1e6,y*1e6,(abs(Ex))); colorbar; shading interp;
        text(textx,texty,'|{\it{E}}_{\it{x}}|','color','w');
        xlabel('{\it{x}} (\mum)');
        ylabel('{\it{y}} (\mum)');
        hold on; contour(x*1e6,y*1e6,abs(eps)*max(max((abs(Ex))))/100,1,'linewidth',1,'linecolor','w','linestyle','--');
        subplot(232);
        pcolor(x*1e6,y*1e6,(abs(Ey))); colorbar; shading interp;
        % % if maxE>1e3
        % %     title('Probably Spurious')
        % % end
        text(textx2,texty2,['n_{eff} = ' num2str(real(neff))],'color','w')
        text(textx,texty,'|{\it{E}}_{\it{y}}|','color','w');
        xlabel('{\it{x}} (\mum)');
        ylabel('{\it{y}} (\mum)');
        hold on; contour(x*1e6,y*1e6,abs(eps)*max(max((abs(Ey))))/100,1,'linewidth',1,'linecolor','w','linestyle','--');
        subplot(233);
        pcolor(x*1e6,y*1e6,(abs(Ez))); colorbar; shading interp;
        text(textx,texty,'|{\it{E}}_{\it{z}}|','color','w');
        xlabel('{\it{x}} (\mum)');
        ylabel('{\it{y}} (\mum)');
        hold on; contour(x*1e6,y*1e6,abs(eps)*max(max((abs(Ez))))/100,1,'linewidth',1,'linecolor','w','linestyle','--');
        subplot(234);
        pcolor(x*1e6,y*1e6,(abs(Hx))); colorbar; shading interp;
        text(textx,texty,'|{\it{H}}_{\it{x}}|','color','w');
        xlabel('{\it{x}} (\mum)');
        ylabel('{\it{y}} (\mum)');
        hold on; contour(x*1e6,y*1e6,abs(eps)*max(max((abs(Hx))))/100,1,'linewidth',1,'linecolor','w','linestyle','--');
        subplot(235);
        pcolor(x*1e6,y*1e6,(abs(Hy))); colorbar; shading interp;
        text(textx,texty,'|{\it{H}}_{\it{y}}|','color','w');
        xlabel('{\it{x}} (\mum)');
        ylabel('{\it{y}} (\mum)');
        hold on; contour(x*1e6,y*1e6,abs(eps)*max(max((abs(Hy))))/100,1,'linewidth',1,'linecolor','w','linestyle','--');
        subplot(236);
        pcolor(x*1e6,y*1e6,(abs(Hz))); colorbar; shading interp;
        text(textx,texty,'|{\it{H}}_{\it{z}}|','color','w');
        xlabel('{\it{x}} (\mum)');
        ylabel('{\it{y}} (\mum)');
        hold on; contour(x*1e6,y*1e6,abs(eps)*max(max((abs(Hz))))/100,1,'linewidth',1,'linecolor','w','linestyle','--');

        if save_plot == 1
            print('-dpng','-r100',[filename '_WG_mode' int2str(mm)])
        end
    end
end