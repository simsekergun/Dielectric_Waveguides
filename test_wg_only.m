clear;
close all;

% tidy3D_results = [1.795121 1.754014 1.648527 1.629351];
% our_results = [    1.7933    1.7507    1.6464   1.6257];
% percentile_errors = [ 0.1014    0.1889    0.1290    0.2241];

%%%%%%%%%%%%%%%%%%%%%%%%%%
set(0,'defaultlinelinewidth',2)
set(0,'DefaultAxesFontSize',18)
set(0,'DefaultTextFontSize',18)
c0 = 3e8;           % speed of EM waves in vacuum

%%% Optical index Geometry %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n_wg = 1.9761;
n_background = 1.444;

filename = 'Waveguide_Only';
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
nmodes_search=20;                           %% number of solutions asked
desired_modes = 1:10;
n_substrate = []; 
n_film = [];
film_thickness = [];
plot_geo = 1;
plot_fields = 1;
save_figure = 1;

%% END OF INPUT Parameters

neff_min = 1.4;
neff_max = 0.99*n_wg;

[x,y,dx,dy,epsr] = get_uniform_mesh(structure,lambda0, wg_width_bottom, wg_height, gap_left, gap_right, gap_top, ...
    gap_bottom,  ppw, plot_geo, n_background, n_wg,n_substrate, n_film, film_thickness, wg_width_top);

[Ex, Ey, Ez, Hx, Hy, Hz, beta, neff, time_spent]=Waveguide_Solver(x,y,dx,dy,epsr,lambda0,desired_modes, neff_min,neff_max, plot_fields,save_figure, filename);

save(filename)
!mv *.mat ./mat_files/
!mv *.png ./png_files/



