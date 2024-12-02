clear;
close all;
tidy3D_results = [1.795121 1.754014 1.648527 1.629351];
ergun_results = [1.7940 1.7515 1.6476 1.6270];
percentile_errors = [0.0624    0.1433    0.0562    0.1443];
[(1:4).' tidy3D_results.' ergun_results.'  percentile_errors.']

% desired_modes = [2 3 5 6];
%%%%%%%%%%%%%%%%%%%%%%%%%%
set(0,'defaultlinelinewidth',2)
set(0,'DefaultAxesFontSize',18)
set(0,'DefaultTextFontSize',18)
c0 = 3e8;           % speed of EM waves in vacuum

%%% Optical index Geometry %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n_wg = 1.9761;
n_background = 1.444;

filename = 'test_wg_only';
lambda0=1550e-9;
structure = 'wg_only';
wg_width_bottom = 1.6e-6;
wg_width_top = wg_width_bottom;
wg_height = 0.7e-6;
gap_left = 1e-6;
gap_right = gap_left;
gap_top = gap_left;
gap_bottom = gap_left;
ppw = 20;
nmodes_search=6;                           %% number of solutions asked
desired_modes = [2 3 5 6];
n_substrate = []; 
n_film = [];
film_thickness = [];
plot_geo = 1;
plot_fields = 1;
save_figure = 0;

%% END OF INPUT Parameters

neff_min = 1;
neff_max = 0.99*n_wg;

[x,y,dx,dy,epsr] = get_mesh(structure,lambda0, wg_width_bottom, wg_height, gap_left, gap_right, gap_top, ...
    gap_bottom,  ppw, plot_geo, n_background, n_wg,n_substrate, n_film, film_thickness, wg_width_top);

[Nz, Nr] = size(epsr);
neffs = zeros(length(desired_modes),1);
betas= zeros(length(desired_modes),1);
Exs = zeros(Nz, Nr, length(desired_modes));
Eys = zeros(Nz, Nr, length(desired_modes));
Ezs = zeros(Nz, Nr, length(desired_modes));
Hxs = zeros(Nz, Nr, length(desired_modes));
Hys = zeros(Nz, Nr, length(desired_modes));
Hzs = zeros(Nz, Nr, length(desired_modes));

for ii = 1:length(desired_modes)
    [Ex, Ey, Ez, Hx, Hy, Hz, beta, neff]=Solver_Waveguide_NU(x,y,dx,dy,epsr,lambda0,nmodes_search,desired_modes(ii), neff_min,neff_max,plot_fields,save_figure, filename);
    neffs(ii) = neff;
    betas(ii) = neff;
    Exs(:,:,ii) = Ex;
    Eys(:,:,ii) = Ey;
    Ezs(:,:,ii) = Ez;
    Hxs(:,:,ii) = Hx;
    Hys(:,:,ii) = Hy;
    Hzs(:,:,ii) = Hz;
end
disp(neffs);

save(filename)
!mv *.mat ./mat_files/
