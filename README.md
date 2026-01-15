# Fractional Compact Schemes

This repository contains implementations of compact finite difference schemes for fractional derivatives, specifically for **Weyl** and **Riesz** fractional derivatives. The code is available in both **MATLAB** and **Python** versions.

## Overview

Fractional derivatives extend the concept of classical integer-order derivatives to non-integer orders. This project implements high-order compact finite difference schemes for approximating fractional derivatives, with a focus on:

- **Weyl Fractional Derivatives**: Right-sided fractional derivatives on infinite or semi-infinite domains
- **Riesz Fractional Derivatives**: Symmetric fractional derivatives
- **Periodic and Non-Periodic Boundary Conditions**: Different boundary treatments for various domain types
- **Frequency Domain Analysis**: Analysis of scheme properties in the frequency domain

## Repository Structure

```
Fractional_compact/
├── (Matlab)Fractional_compact/
│   ├── Weyl_seven_pts(Matlab)/      # Weyl derivative, 7-point compact schemes
│   ├── Weyl_frequency(Matlab)/       # Weyl derivative, frequency domain analysis
│   ├── Riesz_five_pts(Matlab)/       # Riesz derivative, 5-point compact schemes
│   ├── Riesz_four_pts(Matlab)/       # Riesz derivative, 4-point compact schemes
│   └── Riesz_frequency(Matlab)/      # Riesz derivative, frequency domain analysis
└── (Python)Fractional_compact/
    ├── Weyl_seven_pts(python)/       # Python version of Weyl 7-point schemes
    ├── Weyl_frequency(python)/       # Python version of Weyl frequency analysis
    ├── Riesz_five_pts(python)/       # Python version of Riesz 5-point schemes
    ├── Riesz_four_pts(python)/       # Python version of Riesz 4-point schemes
    └── Riesz_frequency(python)/      # Python version of Riesz frequency analysis
```

## Core Concepts

### Compact Finite Difference Schemes

Compact schemes use a combination of function values and their derivatives at neighboring grid points, leading to higher accuracy with the same stencil width compared to traditional finite difference methods.

### Interpolation/Collocation Method

The coefficients of the compact schemes are determined by matching the numerical scheme's frequency response to the exact fractional derivative's frequency response at specific collocation points:

- **Exact frequency response**: `(iω)^γ` for Weyl derivatives
- **Numerical scheme**: A linear combination of basis functions representing the scheme structure
- **Optimization**: Solve a least-squares problem to minimize the error at collocation points

## Main Directories and Files

### 1. Weyl Seven-Point Schemes (`Weyl_seven_pts`)

**Purpose**: 7-point compact schemes for Weyl fractional derivatives with periodic and non-periodic boundary conditions.

#### Core Functions

- **`solve_redu_inter_h.m/py`**
  - Solves the reduced interpolation problem for the internal (periodic) scheme
  - Eliminates parameter `a4` from the 7-point stencil
  - **Inputs**: `gamma` (fractional order), `h` (grid spacing), `x_points_inter` (collocation points, optional), `plot_flag`
  - **Outputs**: Optimized coefficients `[a1, a2, a3, a5, a6, a7, alpha, beta]`, coefficient matrix `A`, RHS vector `b`, L2 error

#### Boundary Schemes

For non-periodic domains, different boundary closure schemes are implemented:

- **`solve_redu_inter_bdy_i1_h.m/py`**: Boundary scheme eliminating `a1`
- **`solve_redu_inter_bdy_i2_h.m/py`**: Boundary scheme eliminating `a2`
- **`solve_redu_inter_bdy_i3_h.m/py`**: Boundary scheme eliminating `a3`
- **`solve_redu_inter_bdy_iN_1_h.m/py`**: Near-boundary scheme eliminating `a4`
- **`solve_redu_inter_bdy_iN_h.m/py`**: Boundary scheme eliminating `a6`
- **`solve_redu_inter_bdy_iN1_h.m/py`**: Boundary scheme eliminating `a7`

Each boundary scheme is designed to maintain the compact structure while handling boundary constraints.

#### Test Files

- **`test_periodic_inter_compact_h.m/py`**
  - Tests the periodic scheme on `u(x) = sin(2πkx)`
  - Performs single tests and convergence studies
  - Plots exact vs. numerical derivatives and error analysis

- **`test_periodic_inter_diff_gamma_compact_h.m/py`**
  - Scans different `gamma` values for fixed grid resolution
  - Plots absolute and relative L∞ errors vs. `gamma`

- **`test_pde_with_periodic_schemes_h.m/py`**
  - Tests 1D PDE: `D^γ u + u = f` with periodic boundary conditions
  - Uses test function `u(x) = sin(2πkx)`
  - Solves the PDE and compares with exact solution

- **`test_pde_2d_periodic_schemes_h.m/py`**
  - Tests 2D PDE with Weyl fractional derivatives and periodic boundary conditions
  - Uses test function `u(x,y) = sin(2πk₁x) sin(2πk₂y)`
  - Visualizes 3D solution surfaces

- **`test_exp_inter_bdy_compact_h.m/py`**
  - Tests non-periodic schemes on `u(x) = exp(λx)` on domain `[-20, 1]`
  - Uses boundary closure schemes (i1, i2, i3, iN_1, iN, iN1)
  - Performs convergence studies and point-wise error analysis

- **`test_exp_inter_bdy_diff_gamma_compact_h.m/py`**
  - Scans different `gamma` values for fixed `N` (grid points)
  - Tests non-periodic schemes with exponential test functions
  - Plots error trends vs. `gamma`

- **`test_pde_nonperi_schemes_h.m/py`** / **`test_pde_nonperiodic_scheme_h.py`**
  - Tests 1D PDE with non-periodic boundary conditions
  - Uses appropriate boundary closure schemes

### 2. Weyl Frequency Domain Analysis (`Weyl_frequency`)

**Purpose**: Frequency domain analysis of the compact schemes, comparing the numerical frequency response with the exact `(iω)^γ`.

#### Core Functions

- **`solve_redu_inter_frequency_h.m/py`**
  - Solves the reduced interpolation problem with frequency domain visualization
  - Plots real/imaginary parts of frequency response and error magnitudes
  - Analyzes scheme accuracy across the frequency spectrum
  - **Key difference**: Focuses on frequency response matching rather than spatial accuracy

#### Boundary Frequency Analysis

- **`solve_redu_inter_bdy_frequency_i1_h.m/py`**: Frequency analysis for boundary scheme i1
- **`solve_redu_inter_bdy_frequency_i2_h.m/py`**: Frequency analysis for boundary scheme i2
- **`solve_redu_inter_bdy_frequency_i3_h.m/py`**: Frequency analysis for boundary scheme i3

These functions visualize how each boundary scheme approximates the exact frequency response `(iω)^γ` in the frequency domain.

### 3. Riesz Schemes (`Riesz_five_pts`, `Riesz_four_pts`)

**Purpose**: Compact schemes for Riesz fractional derivatives (symmetric fractional derivatives).

#### Core Functions

- **`solve_redu_inter_five_h.m/py`**: 5-point compact scheme for Riesz derivatives
- **`solve_redu_inter_four_h.m/py`**: 4-point compact scheme for Riesz derivatives

#### Test Files

Similar structure to Weyl schemes:
- `test_periodic_inter_compact_h.m/py`: Periodic scheme tests
- `test_periodic_inter_diff_gamma_compact_h.m/py`: Error analysis vs. `gamma`
- `test_pde_with_periodic_schemes_h.m/py`: PDE tests with periodic BCs
- `test_pde_2d_periodic_schemes_h.m/py`: 2D PDE tests

### 4. Riesz Frequency Domain Analysis (`Riesz_frequency`)

**Purpose**: Frequency domain analysis for Riesz fractional derivative schemes.

- **`solve_redu_inter_five_frequency_h.m/py`**: Frequency analysis for 5-point Riesz scheme
- **`solve_redu_inter_four_frequency_h.m/py`**: Frequency analysis for 4-point Riesz scheme

## Mathematical Background

### Weyl Fractional Derivative

The right-sided Weyl fractional derivative of order `gamma` is defined as:

```
D^gamma u(x) = (d/dx)^gamma u(x)  (right-sided)
```

In the frequency domain, it corresponds to multiplication by `(iω)^gamma`, where:
- `(iω)^gamma = |ω|^gamma exp(igammaπ/2 sign(ω))`
- Real part: `|ω|^gamma cos(gammaπ/2)`
- Imaginary part: `|ω|^gamma sin(gammaπ/2) sign(ω)`

### Riesz Fractional Derivative

The Riesz fractional derivative is symmetric and defined as:

```
D^γ u(x) = -1/(2 cos(γπ/2)) [D_+^γ u(x) + D_-^γ u(x)]
```

where `D_+^γ` and `D_-^γ` are left and right fractional derivatives.

### Compact Scheme Structure

The 7-point compact scheme approximates `D^γ u` at point `j` as:

```
∑_{k=-3}^{3} a_k u_{j+k} = α D^γ u_{j-1} + β D^γ u_j + α D^γ u_{j+1}
```

with constraint: `∑_{k=-3}^{3} a_k = 0` (conservation property).

The reduced system eliminates one parameter (typically `a4`) and solves for the remaining coefficients.

## Usage Examples

### MATLAB

```matlab
% Periodic scheme for Weyl derivative
gamma = 0.5;
h = 1/64;
[c, A, b, err] = solve_redu_inter_h(gamma, h, [], true);

% Test on periodic function
M = 64;
k_wave = 1;
test_periodic_inter_compact_h(gamma, M, k_wave);

% Frequency domain analysis
N = 64;
solve_redu_inter_frequency_h(gamma, N, [], true);
```

### Python

```python
from solve_redu_inter_h import solve_redu_inter_h
from test_periodic_inter_compact_h import test_periodic_inter_compact_h

# Periodic scheme for Weyl derivative
gamma = 0.5
h = 1/64
c, A, b, err = solve_redu_inter_h(gamma, h, None, True)

# Test on periodic function
M = 64
k_wave = 1
test_periodic_inter_compact_h(gamma, M, k_wave)

# Frequency domain analysis
from solve_redu_inter_frequency_h import solve_redu_inter_frequency_h
N = 64
solve_redu_inter_frequency_h(gamma, N, None, True)
```

## Dependencies

### MATLAB
- MATLAB R2016b or later
- No additional toolboxes required (uses only built-in functions)

### Python
- Python 3.7+
- NumPy
- SciPy (for sparse matrix operations)
- Matplotlib (for plotting)

Install Python dependencies:
```bash
pip install numpy scipy matplotlib
```

## Key Features

1. **High Accuracy**: Compact schemes provide higher accuracy than traditional finite differences with the same stencil width
2. **Frequency Domain Matching**: Coefficients are optimized to match the exact frequency response
3. **Flexible Boundary Conditions**: Multiple boundary closure schemes for non-periodic domains
4. **Comprehensive Testing**: Extensive test suites for both periodic and non-periodic cases
5. **Visualization**: Built-in plotting capabilities for error analysis and frequency response

## File Naming Convention

- **`solve_redu_inter_*`**: Core functions that solve the reduced interpolation problem
- **`test_*`**: Test and validation scripts
- **`*_frequency_*`**: Frequency domain analysis functions
- **`*_bdy_*`**: Boundary scheme functions
- **`*_compact_h`**: Compact scheme functions (h = grid spacing)

## Error Metrics

The code computes various error metrics:

- **L² Error**: `||exact - numerical||₂`
- **L∞ Error**: `max|exact - numerical|`
- **Relative L∞ Error**: `max|exact - numerical| / (1 + |exact|)`
- **Point-wise Error**: Error at each grid point for detailed analysis

## Notes

- Grid spacing `h` and number of points `M`/`N` are related by `h = 1/M` or `h = domain_length/N`
- Collocation points are automatically selected based on `h` if not provided
- Frequency domain analysis focuses on `ω ∈ [0, pi]` (Nyquist frequency)
- Boundary schemes use different parameter elimination strategies to maintain accuracy near boundaries

## License



## Citation

If you use this code in your research, please cite:



## Contact

Yingli.li@colostate.edu
