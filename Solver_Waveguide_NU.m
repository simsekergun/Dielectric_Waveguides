function [Ex, Ey, Ez, Hx, Hy, Hz, beta, neff]=Solver_Waveguide_NU(x,y,dx, dy, eps,lambda,nmodes,desired_mode, neff_min,neff_max, plot_fields,save_plot, filename)
Nx=length(x);
Ny=length(y);
Nxy=Nx*Ny;

k0=2*pi/lambda;
beta_trial = 0.9*k0*neff_max;

f = 3e8/lambda;
w = 2*pi*f;
mu0 = 4e-7*pi;
h_factor = 1i/w/mu0;

%%% Building of the operators
AA=ones(1,Nx*Ny);
BB=ones(1,Nx*Ny-1);
BB(Ny:Ny:end)=0;
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

dxs = repmat(dx,Ny,1);
ddxs = spdiags(1./dxs(:), 0, Nxy, Nxy);     % note that this is 1/dx
dys = repmat(dy.',1,Nx);
ddys = spdiags(1./dys(:), 0, Nxy, Nxy);     % note that this is 1/dy

DX1 = DX1*ddxs;
DY1 = DY1*ddys;
DX2 = DX2*ddxs*ddxs;
DY2 = DY2*ddys*ddys;

deps_dx_inveps= [zeros(Ny,2) -eps(:,1:end-4)+8*eps(:,2:end-3)-8*eps(:,4:end-1) + eps(:,5:end) zeros(Ny,2) ]./eps./(12*dxs);
deps_dy_inveps = [zeros(2,Nx); -eps(1:end-4, :)+8*eps(2:end-3, :)-8*eps(4:end-1, :) + eps(5:end, :); zeros(2,Nx) ]./eps ./ (12*dys);
depsdx_inveps = spdiags(deps_dx_inveps(:), 0, Nxy, Nxy);
depsdy_inveps = spdiags(deps_dy_inveps(:), 0, Nxy, Nxy);


%% Building of the Hamiltonian
eps_diag = spdiags(eps(:), 0, Nxy, Nxy);
error = 1;
while error > 0.01
H9 = DX2 + DY2 + eps_diag * k0^2;
H1 = H9 + DX1 * depsdx_inveps;
H2 = DX1 * depsdy_inveps;
H3 = zeros(size(H9));
H4 = DY1 * depsdx_inveps;
H5 = H9 +DY1 * depsdy_inveps;
H6 = zeros(size(H9));
H7 =  -1i*beta_trial*depsdx_inveps;
H8 =  -1i*beta_trial*depsdy_inveps;

[psi,BetaSquare] = eigs(sparse([H1 H2 H3; H4 H5 H6; H7 H8 H9]),nmodes,'LR');   %% eigen values are ordered
neff=sqrt(diag(BetaSquare))/k0;
%% Filtering and reshaping the Wavefunction
idx1=real(neff)>neff_min;
idx2=real(neff)<neff_max;
idx=logical( idx1.*idx2);

neffs=neff(idx);
betas = k0*neffs;
new_beta = real(betas(desired_mode));
error = abs((new_beta-beta_trial)/new_beta);
beta_trial = new_beta;
end

beta = beta_trial;
neff = beta/k0;

Ex0=psi(1:Nxy,desired_mode);
Ey0=psi(Nxy+1:2*Nxy,desired_mode);
Ez0=psi(2*Nxy+1:end,desired_mode);

Hx = h_factor* reshape( DY1*Ez0+1i*beta_trial*Ey0,[Ny,Nx]);
Hy = h_factor* reshape(-DX1*Ez0-1i*beta_trial*Ex0,[Ny,Nx]);
Hz = h_factor* reshape( DX1*Ey0-DY1*Ex0,[Ny,Nx]);

Ex = reshape(Ex0,[Ny,Nx]);
Ey = reshape(Ey0,[Ny,Nx]);
Ez = reshape(Ez0,[Ny,Nx]);

Power_z_sqrt = sqrt(real(sum(sum(  (Ex(:).*conj(Hy(:))-Ey(:).*conj(Hx(:))).*dxs(:).*dys(:)  )))/lambda^2);

Ex = Ex/Power_z_sqrt;
Ey = Ey/Power_z_sqrt;
Ez = Ez/Power_z_sqrt;
Hx = Hx/Power_z_sqrt;
Hy = Hy/Power_z_sqrt;
Hz = Hz/Power_z_sqrt;

maxE = max([max(max(abs(Ex))) max(max(abs(Ey))) max(max(abs(Ez)))]);
sumEs = [sum(sum(abs(Ex).^2)) sum(sum(abs(Ey).^2)) sum(sum(abs(Ez).^2))];
sumHs = [sum(sum(abs(Hx).^2)) sum(sum(abs(Hy).^2)) sum(sum(abs(Hz).^2))];
Efractions = sumEs./sum(sumEs);
Hfractions = sumHs./sum(sumHs);
disp(['E fractions: ' num2str(Efractions)]);
disp(['H fractions: ' num2str(Hfractions)]);

if plot_fields== 1
    textx = 1e6*(min(x)+ (max(x)-min(x))/15);
    texty = 1e6*(max(y) - (max(y)-min(y))/10);
    textx2 = 1e6*(min(x)+ (max(x)-min(x))/3);
    texty2 = 1e6*(min(y) + (max(y)-min(y))/10);
    
    figure('position',[400, 600, 1200, 500],'Name','WG_Solver');
    subplot(231);
        pcolor(x*1e6,y*1e6,(abs(Ex))); colorbar; shading interp;
        text(textx,texty,'|E_x|','color','w');
        xlabel('x (\mum)');
        ylabel('y (\mum)');
        hold on; contour(x*1e6,y*1e6,abs(eps)*max(max((abs(Ex))))/100,1,'linewidth',1,'linecolor','w','linestyle','--');
    subplot(232);
        pcolor(x*1e6,y*1e6,(abs(Ey))); colorbar; shading interp;
        if maxE>1e3
            title('Probably Spurious')
        end
        text(textx2,texty2,['n_{eff} = ' num2str(real(neff))],'color','w')
        text(textx,texty,'|E_y|','color','w');
        xlabel('x (\mum)');
        ylabel('y (\mum)');
        hold on; contour(x*1e6,y*1e6,abs(eps)*max(max((abs(Ey))))/100,1,'linewidth',1,'linecolor','w','linestyle','--');
    subplot(233);
        pcolor(x*1e6,y*1e6,(abs(Ez))); colorbar; shading interp;        
        text(textx,texty,'|E_z|','color','w');
        xlabel('x (\mum)');
        ylabel('y (\mum)');
        hold on; contour(x*1e6,y*1e6,abs(eps)*max(max((abs(Ez))))/100,1,'linewidth',1,'linecolor','w','linestyle','--');
    subplot(234);
        pcolor(x*1e6,y*1e6,(abs(Hx))); colorbar; shading interp;
        text(textx,texty,'|H_x|','color','w');
        xlabel('x (\mum)');
        ylabel('y (\mum)');
        hold on; contour(x*1e6,y*1e6,abs(eps)*max(max((abs(Hx))))/100,1,'linewidth',1,'linecolor','w','linestyle','--');
    subplot(235);
        pcolor(x*1e6,y*1e6,(abs(Hy))); colorbar; shading interp;        
        text(textx,texty,'|H_y|','color','w');
        xlabel('x (\mum)');
        ylabel('y (\mum)');
        hold on; contour(x*1e6,y*1e6,abs(eps)*max(max((abs(Hy))))/100,1,'linewidth',1,'linecolor','w','linestyle','--');
    subplot(236);
        pcolor(x*1e6,y*1e6,(abs(Hz))); colorbar; shading interp;        
        text(textx,texty,'|H_z|','color','w');
        xlabel('x (\mum)');
        ylabel('y (\mum)');
        hold on; contour(x*1e6,y*1e6,abs(eps)*max(max((abs(Hz))))/100,1,'linewidth',1,'linecolor','w','linestyle','--');

        if save_plot == 1
            print('-dpng','-r100',[filename '_WG_mode' int2str(desired_mode)])
        end
end
