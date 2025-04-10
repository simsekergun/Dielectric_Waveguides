clear;
close all;

% tidy3D_results = [1.889322 1.834376	1.765082	1.682733];
% our_results = [1.8890 1.8230 1.7497 1.6864];
% percentile_errors = 100*abs(tidy3D_results-our_results)./tidy3D_results;
% percentile_errors = 0.0170    0.6202    0.8715    0.2179

%%%%%%%%%%%%%%%%%%%%%%%%%%
set(0,'defaultlinelinewidth',2)
set(0,'DefaultAxesFontSize',18)
set(0,'DefaultTextFontSize',18)
c0 = 3e8;           % speed of EM waves in vacuum

%%% Optical index Geometry %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n_wg = 2.1;
n_background = 1;
n_substrate = 1.6;
n_film = n_wg;


filename = 'test_tapered_wg_on_film_and_substrate';
lambda0=1550e-9;
structure = 'tapered_wg_on_film_and_substrate';
wg_width_bottom = 1.2e-6;
wg_width_top = 1.0e-6;
wg_height = 0.35e-6;
gap_left = 1e-6;
gap_right = gap_left;
gap_top = gap_left;
gap_bottom = gap_left;
ppw = 30;
desired_modes = 1:4;
film_thickness = 0.35e-6;
plot_geo = 1;
plot_fields = 1;
save_figure = 0;

%% END OF INPUT Parameters

neff_min = 1;
neff_max = 0.99*n_wg;

[x,y,dx,dy,epsr] = get_uniform_mesh(structure,lambda0, wg_width_bottom, wg_height, gap_left, gap_right, gap_top, ...
    gap_bottom,  ppw, plot_geo, n_background, n_wg,n_substrate, n_film, film_thickness, wg_width_top);

[Exs, Eys, Ezs, Hxs, Hys, Hzs, betas, neffs]=Waveguide_Solver(x,y,dx,dy,epsr,lambda0,desired_modes, neff_min,neff_max, plot_fields,save_figure, filename);

save(filename)
!mv *.mat ./mat_files/



