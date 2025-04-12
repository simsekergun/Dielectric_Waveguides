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

# Waveguide Mode Analysis Formulation

The formulation starts with the vector wave equation for the electric field

$$
\nabla^2\mathbf{E} + \nabla\left(\frac{1}{\varepsilon}\nabla\varepsilon \cdot \mathbf{E}\right) + k_0^2\varepsilon_r\mathbf{E} = 0,
$$  
where $k_0 = \omega\sqrt{\mu_0\varepsilon_0}$ is the free-space wavenumber and $\varepsilon_r = \varepsilon/\varepsilon_0$ is the relative electrical permittivity.

## Field Decomposition

We employ phasor notation and consider a two-dimensional configuration by assuming that the waveguide extends infinitely along the $z$-axis. Accordingly, we express the electric field as  

$$
\mathbf{E}(x,y,z) = \left[ \hat{x}E_x(x,y) + \hat{y}E_y(x,y) + \hat{z}E_z(x,y) \right] e^{-j\beta z},
$$  
where $\beta$ denotes the propagation constant in the $z$-direction. By substituting the field expression into the wave equation, we derive the coupled equations governing the components $E_x$, $E_y$, and $E_z$.

## Expanded Wave Equation Terms

The full form of the term $\nabla \left( \frac{1}{\varepsilon} \nabla \varepsilon \cdot \mathbf{E} \right)$ is:

$$
\begin{aligned}
\nabla \left( \frac{1}{\varepsilon} \nabla \varepsilon \cdot \mathbf{E} \right) = & \left\{
\hat{\mathbf{x}} \frac{\partial}{\partial x} \left[ \frac{1}{\varepsilon} \left( \frac{\partial \varepsilon}{\partial x} E_x + \frac{\partial \varepsilon}{\partial y} E_y \right) \right] \right. \\
& + \hat{\mathbf{y}} \frac{\partial}{\partial y} \left[ \frac{1}{\varepsilon} \left( \frac{\partial \varepsilon}{\partial x} E_x + \frac{\partial \varepsilon}{\partial y} E_y \right) \right] \\
& \left. + \hat{\mathbf{z}} \left[ -j\beta \frac{1}{\varepsilon} \left( \frac{\partial \varepsilon}{\partial x} E_x + \frac{\partial \varepsilon}{\partial y} E_y \right) \right] \right\} e^{-j\beta z}.
\end{aligned}
$$

## Scalar Wave Equations

The resulting wave equation decomposes into three scalar equations:

1. $x$-component:
$$
\frac{\partial^2 E_x}{\partial x^2} + \frac{\partial^2 E_x}{\partial y^2} 
+ \frac{\partial}{\partial x} \frac{1}{\varepsilon} \left( \frac{\partial \varepsilon}{\partial x} E_x + \frac{\partial \varepsilon}{\partial y} E_y \right)
+ k_0^2\varepsilon_r E_x = \beta^2 E_x
$$

2. $y$-component:
$$
\frac{\partial^2 E_y}{\partial x^2} + \frac{\partial^2 E_y}{\partial y^2} 
+ \frac{\partial}{\partial y} \frac{1}{\varepsilon} \left( \frac{\partial \varepsilon}{\partial x} E_x + \frac{\partial \varepsilon}{\partial y} E_y \right) 
+ k_0^2\varepsilon_r E_y = \beta^2 E_y
$$

3. $z$-component:
$$
\frac{\partial^2 E_z}{\partial x^2} + \frac{\partial^2 E_z}{\partial y^2}   
-j\beta \frac{1}{\varepsilon} \left( \frac{\partial \varepsilon}{\partial x} E_x + \frac{\partial \varepsilon}{\partial y} E_y \right)
+ k_0^2\varepsilon_r E_z = \beta^2 E_z
$$

## Alternative Formulation

The $z$-component can alternatively be determined from:
$$
E_z = \frac{1}{j\beta \varepsilon_r} \left( \frac{\partial}{\partial x} \varepsilon_r E_x + \frac{\partial}{\partial y} \varepsilon_r E_y \right)
$$

However, this formulation may introduce numerical inaccuracies at material interfaces, motivating our full three-component approach.

## Matrix Formulation

The equations can be cast in matrix form:

$$
\begin{bmatrix}
M_1 & M_2 & M_3 \\
M_4 & M_5 & M_6 \\
M_7 & M_8 & M_9
\end{bmatrix}
\begin{bmatrix}
E_x \\
E_y \\
E_z 
\end{bmatrix}
= 
\beta^2 
\begin{bmatrix}
E_x \\
E_y \\
E_z 
\end{bmatrix}
$$

where $M_3 = M_6 =0$ and:

$$
\begin{aligned}
M_1 & = M_9 + \frac{\partial}{\partial x} \frac{1}{\varepsilon} \frac{\partial \varepsilon}{\partial x} \\
M_2 & = \frac{\partial}{\partial x} \frac{1}{\varepsilon} \frac{\partial \varepsilon}{\partial y} \\
M_4 & = \frac{\partial}{\partial y} \frac{1}{\varepsilon} \frac{\partial \varepsilon}{\partial x} \\
M_5 & = M_9 + \frac{\partial}{\partial y} \frac{1}{\varepsilon} \frac{\partial \varepsilon}{\partial y} \\
M_7 & = -j\beta \frac{1}{\varepsilon} \frac{\partial \varepsilon}{\partial x} \\
M_8 & = -j\beta \frac{1}{\varepsilon} \frac{\partial \varepsilon}{\partial y} \\
M_9 & = \frac{\partial^2 }{\partial x^2} + \frac{\partial^2}{\partial y^2} + k_0^2\varepsilon_r
\end{aligned}
$$

## Magnetic Field Components

The magnetic field components can be found from:

$$
\begin{aligned}
H_x &= \frac{j}{\omega \mu_0} \left(\frac{\partial E_z}{\partial y} + j\beta E_y\right) \\
H_y &= \frac{j}{\omega \mu_0} \left( -j\beta E_x - \frac{\partial E_z}{\partial x} \right) \\
H_z &= \frac{j}{\omega \mu_0} \left( \frac{\partial E_y}{\partial x} - \frac{\partial E_x}{\partial y} \right)
\end{aligned}
$$

## Generalized Eigenvalue Problem

The system is reformulated as:

$$
\overline{\overline{{M}}}\mathbf{E} + \beta \overline{\overline{{L}}}\mathbf{E} - \beta^2 \overline{\overline{{I}}} \mathbf{E} = 0
$$

leading to the augmented generalized eigenvalue problem:

$$
\begin{pmatrix}
\overline{\overline{{M}}} & \overline{\overline{{L}}} \\
0 & \overline{\overline{{I}}}
\end{pmatrix}
\begin{pmatrix}
\mathbf{E} \\
\mathbf{F}
\end{pmatrix}
= \beta
\begin{pmatrix}
0 & \overline{\overline{{I}}} \\
\overline{\overline{{I}}} & 0
\end{pmatrix}
\begin{pmatrix}
\mathbf{E} \\
\mathbf{F}
\end{pmatrix}
$$

where $\mathbf{F} = \beta \mathbf{E}$.
