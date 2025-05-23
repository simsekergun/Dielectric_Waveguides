# Dielectric_Waveguides
A finite-difference-based MATLAB solver to study light propagation along straight dielectric waveguides

## MATLAB codes
test_wg_only.m: Si3N4 rectangular waveguide surrounded by SiO2

test_tapered_wg_on_film_substrate.m: thin film lithium niobate waveguide on a substrate

get_uniform_mesh.m: a simple mesh generator. supported shapes
rectangular waveguide in a homogeneous background
rectangular waveguide on top of a substrate
rectangular waveguide on top of a film-coated substrate
tapered waveguide in a homogeneous background
tapered waveguide on top of a substrate
tapered waveguide on top of a film-coated substrate

Waveguide_Solver.m: FD solver for dielectric waveguides

## Notebook
WG_ModeSolver_Paper_Tidy3D_Tests.ipynb: analyses of the waveguides with a commercial solver, Tidy3D.

## Formulation
[Draft](https://arxiv.org/abs/2503.17746)
