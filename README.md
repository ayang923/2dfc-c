# 2DFC-corners-C

> A C implementation of the 2D Fourier Continuation algorithm for domains bounded by C² curves with corners.

---

## Table of Contents

- [Overview](#overview)
- [Setup](#setup)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Building](#building)
- [Repository Structure](#repository-structure)
- [How to Run the 2DFC Algorithm](#how-to-run-the-2dfc-algorithm)
  1. [Obtain FC Matrices and Select Parameters](#1-obtain-fc-matrices-and-select-parameters)
  2. [Define the Domain via `curve_seq_t`](#2-define-the-domain-via-curve_seq_t)
  3. [Call `FC2D`](#3-call-fc2d)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

---

## Overview

This repository is a C implementation of the **2D Fourier Continuation (2DFC)** algorithm, which takes a function $f(x,y)$ defined on an arbitrary 2D domain bounded by a sequence of $C^2$ curves and produces a smooth, periodic Fourier series representation on a bounding Cartesian rectangle.  This allows spectral-accuracy computations (derivatives, Poisson solvers, etc.) on non-rectangular domains.

The algorithm is described in detail in:

> Bruno, Oscar P., and Allen Yang. Two-Dimensional Fourier Continuation for Domains with Corners. Submitted manuscript, 2026.

A MATLAB reference implementation is available at [ayang923/2dfc-matlab](https://github.com/ayang923/2dfc-matlab).

---

## Setup

### Requirements

- **Intel oneAPI Base Toolkit** — provides the `icx` C compiler.
- **Intel oneAPI Math Kernel Library (MKL)** — provides BLAS, LAPACK, and FFT routines.
- **MATLAB** (optional) — required only if you need to generate FC matrices for new `(d, C, n_r)` parameters (see [Section 1](#1-obtain-fc-matrices-and-select-parameters)).

### Installation

Install the Intel oneAPI compiler and MKL on a Debian/Ubuntu system:

```bash
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
    | gpg --dearmor \
    | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
echo "deb https://apt.repos.intel.com/oneapi all main" \
    | sudo tee /etc/apt/sources.list.d/intel-oneapi.list
sudo apt update
sudo apt install -y intel-oneapi-compiler-dpcpp-cpp intel-oneapi-mkl-devel
```

Activate the environment (add to `~/.bashrc` to make it persistent):

```bash
source /opt/intel/oneapi/setvars.sh
```

> **Note:** The exact path to `setvars.sh` may vary depending on your installation. Check under `/opt/intel/oneapi/` if the path above does not exist.

### Building

From the repository root:

```bash
make
```

This compiles all library objects under `out/` and links the example binary under `bin/`. To build a specific example, edit the `TARGETS` variable in the `Makefile`. To clean all build artifacts:

```bash
make clean
```

---

## Repository Structure

```
2dfc-c/
├── README.md                        This guide
├── Makefile
├── fc_data/                         Precomputed FC matrices (text files)
│   ├── A_d{d}_C{C}_r{n_r}.txt
│   └── Q_d{d}_C{C}_r{n_r}.txt
├── include/                         Public header files
│   ├── fc2D_lib.h                   Main 2DFC entry point
│   ├── curve_seq_lib.h              Boundary curve sequence and patch construction
│   ├── s_patch_lib.h                Smooth boundary patch (S-patch)
│   ├── c_patch_lib.h                Corner patch (C1/C2)
│   ├── q_patch_lib.h                Base quadrilateral parametric patch
│   ├── fc_lib.h                     1D FC gram-blend and matrix I/O
│   ├── r_cartesian_mesh_lib.h       Cartesian output mesh and error computation
│   └── num_linalg_lib.h             Matrix types, interpolation, and utilities
├── src/                             Library source files
│   ├── fc2D_lib.c
│   ├── curve_seq_lib.c
│   ├── s_patch_lib.c
│   ├── c_patch_lib.c
│   ├── q_patch_lib.c
│   ├── fc_lib.c
│   ├── r_cartesian_mesh_lib.c
│   └── num_linalg_lib.c
└── examples/
    ├── boomerang_2DFC.c             Single smooth closed curve
    ├── teardrop_2DFC.c              Teardrop with moderate tip
    ├── teardrop_sharp_2DFC.c        Very sharp teardrop (near-cusp)
    ├── heart_sharp_2DFC.c           Heart with near-cusp indentation
    └── guitarbase_2DFC.c            Four-curve guitar-body domain
```

---

## How to Run the 2DFC Algorithm

Running the 2DFC algorithm requires three steps:

1. Obtain the FC matrices for the chosen parameters and load them from disk.
2. Define the domain boundary as a sequence of $C^2$ curves using `curve_seq_t`.
3. Call `FC2D` (or `FC2D_heap` for large problems).

### 1. Obtain FC Matrices and Select Parameters

#### Parameter Reference

**Function and mesh parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `f` | `scalar_func_2D_t` | Function `f(x, y)` to be represented |
| `h` | `double` | Cartesian mesh step size |

**1D-FC parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `d` | `MKL_INT` | 1D-BTZ matching mesh size; controls accuracy. Typical values: 6–10 |
| `C` | `MKL_INT` | Number of continuation points |
| `n_r` | `MKL_INT` | Refinement factor: the continuation grid is `n_r` times finer than the patch mesh |

**Interpolation and Newton solver parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `M` | `MKL_INT` | Barycentric interpolation stencil width; `d + 3` is a good default |
| `eps_xi_eta` | `double` | Newton solver tolerance in parameter (ξ, η) space; `1e-13` is typical |
| `eps_xy` | `double` | Newton solver tolerance in physical (x, y) space; set equal to `eps_xi_eta` in most cases |

**FFT grid parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `n_x_fft` | `MKL_INT` | FFT grid width; pass `-1` to use the natural Cartesian mesh width |
| `n_y_fft` | `MKL_INT` | FFT grid height; pass `-1` to use the natural Cartesian mesh height |

Passing explicit `n_x_fft` / `n_y_fft` values larger than the natural grid size pads the FFT, which can be useful for selecting highly composite grid sizes that are efficient for the FFT.

#### Obtaining FC Matrices

The FC matrices `A` and `Q` encode the 1D continuation operation and must match the chosen `(d, C, n_r)` triple.  Precomputed matrices for common parameters are stored in `fc_data/` as plain-text files.

If you need matrices for a new `(d, C, n_r)` combination, generate them using the MATLAB repository [ayang923/2dfc-matlab](https://github.com/ayang923/2dfc-matlab):

```matlab
% In MATLAB, from the 2dfc-matlab repository root:
generate_bdry_continuations(d, C, C, 12, 20, 4, 256, n_r);
```

This writes `A_d{d}_C{C}_r{n_r}.txt` and `Q_d{d}_C{C}_r{n_r}.txt` to `data/FC_data/`. Copy those files into the `fc_data/` directory of this repository.

#### Loading FC Matrices in C

```c
MKL_INT d = 7, C = 27, n_r = 6;
MKL_INT M = d + 3;

double A_data[fc_A_numel(d, C, n_r)];
double Q_data[fc_Q_numel(d)];
rd_mat_t A = rd_mat_init_no_shape(A_data);
rd_mat_t Q = rd_mat_init_no_shape(Q_data);

char A_fp[100], Q_fp[100];
sprintf(A_fp, "fc_data/A_d%d_C%d_r%d.txt", d, C, n_r);
sprintf(Q_fp, "fc_data/Q_d%d_C%d_r%d.txt", d, C, n_r);
read_fc_matrix(d, C, n_r, A_fp, Q_fp, &A, &Q);
```

---

### 2. Define the Domain via `curve_seq_t`

#### Background

The domain boundary must be expressible as an ordered, counter-clockwise sequence of $k$ $C^2$ curves $(c_1, c_2, \dots, c_k)$ such that the end of each curve meets the start of the next:

$$(\ell_1^i(1),\, \ell_2^i(1)) = (\ell_1^{i+1}(0),\, \ell_2^{i+1}(0)) \quad \text{for } i = 1, \dots, k$$

Each curve is parametrized as $(x, y) = (\ell_1(\theta),\, \ell_2(\theta))$ for $\theta \in [0,1]$.

#### Patches and Why They Are Needed

The 2DFC algorithm requires the function to be extended smoothly to zero near the boundary. It does this by tiling the boundary region with overlapping **patches** — small curvilinear coordinate systems in which a smooth blending-to-zero extension can be applied.

Three types of patches are constructed per curve:

- **S-patch (smooth patch):** covers most of the curve's interior region, away from the junctions at $\theta=0$ and $\theta=1$.
- **Corner patch:** centered at each junction point ($\theta=0$ of the current curve / $\theta=1$ of the previous curve, and vice versa). Corner patches handle the angular geometry at curve junctions, which the S-patch cannot. C1-type patches handle concave corners (interior angle > 180°); C2-type patches handle convex corners (interior angle < 180°). Corner type is detected automatically from the cross-product of adjacent curve tangent vectors.

The S-patch and corner patches overlap, and a **partition of unity** blends their contributions smoothly in the overlapping region.

> **Remark (convex C2 patches).** The **C2-type** (convex corner) patch construction and its partition-of-unity weights assume, for implementation simplicity, that **neighboring patches do not intersect** the region $\mathcal{M}_p^{\mathcal{C}_2}\bigl((d{-}1)h_\xi^{\mathcal{C}_2} \times (d{-}1)h_\eta^{\mathcal{C}_2}\bigr)$ — the image in $(x,y)$ of the product $(d{-}1)h_\xi^{\mathcal{C}_2} \times (d{-}1)h_\eta^{\mathcal{C}_2}$ in $(\xi,\eta)$ under the parametrization $\mathcal{M}_p^{\mathcal{C}_2}$ (notations as in the paper). In practice, tune `frac_n_*` and `h_norm` so adjacent S/C1 patches avoid overlapping that set.

#### Patch Construction Parameters

To simplify patch construction, `curve_seq_add_curve` accepts fractional parameters that express patch sizes as fractions of the curve's total discretization count `n`, making them scale-invariant. Each curve is first **uniformly discretized** in parameter space: $\theta \in [0,1]$ is sampled at `n` equally spaced points (or `n` is chosen automatically from arc length and `h_norm`). All patch widths are then expressed as **fractions of `n`**, so the same relative layout works when you refine the mesh.

- A **fraction of the `n` points** (given by `frac_n_C_0`) is allocated to the **corner patch centered at $\theta = 0$** — that patch uses the discretization near the start of the curve.
- Another **fraction** (`frac_n_C_1`) is allocated to the **corner patch centered at $\theta = 1$** — the discretization near the end of the curve.
- The remaining interior segment supports the **S-patch**; the `frac_n_S_0` and `frac_n_S_1` parameters then describe how the S-patch **overlaps** into each corner patch (for the partition of unity), again as fractions relative to the corner-patch segment sizes.

This "count points along $\theta$, then carve out corner regions by fractions" design keeps the interface simple and **scale-invariant**: you tune relative patch sizes without re-specifying physical lengths whenever `h` changes.

Each call to `curve_seq_add_curve` accepts:

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | `MKL_INT` | **Uniform $\theta$-grid size.** Number of equally spaced samples of $\theta \in [0,1]$ (endpoints included), i.e. $n-1$ intervals in $\theta$. Pass `0` to auto-set `n = ceil(arc_length / h_norm) + 1`. |
| `frac_n_C_0` | `double` | **Corner patch at $\theta=0$.** Fraction of `n` used for the corner patch at the start of this segment (junction with the previous curve). Counts: `n_C_0 = ceil(frac_n_C_0 * n)` $\theta$-samples from the low-$\theta$ side. Pass `0` for default `ceil(n/10)`. |
| `frac_n_C_1` | `double` | **Corner patch at $\theta=1$.** Fraction of `n` for the corner patch at the end of this segment (junction with the next curve). Counts: `n_C_1 = ceil(frac_n_C_1 * n)` samples from the high-$\theta$ side. Pass `0` for default `ceil(n/10)`. |
| `frac_n_S_0` | `double` | **S-patch overlap at the $\theta=0$ corner.** Fraction of `n_C_0` (not of `n`) by which the smooth patch extends into the $\theta=0$ corner patch: `n_S_0 = ceil(frac_n_S_0 * n_C_0)` — this sets the partition-of-unity overlap width. Pass `0` for default `ceil(2/3 * n_C_0)`. |
| `frac_n_S_1` | `double` | **S-patch overlap at the $\theta=1$ corner.** Fraction of `n_C_1`: `n_S_1 = ceil(frac_n_S_1 * n_C_1)`. Pass `0` for default `ceil(2/3 * n_C_1)`. |
| `h_norm` | `double` | **Normal-direction resolution.** Physical step size in $(x,y)$ for the patch mesh inward from the boundary; with `n` it determines how the boundary strip is sampled perpendicular to the curve. |

Larger `frac_n_C_*` gives wider corner patches along $\theta$; larger `frac_n_S_*` widens the S-patch's overlap into each corner patch (stronger S contribution in the blend).

> **More control:** The `curve_seq_add_curve` interface is intentionally simple. If you need finer control — e.g., setting exact arc-length widths or non-uniform discretizations — you can initialize `s_patch_t` and `c_patch_t` objects and call the patch construction functions directly. This is more involved but gives complete flexibility over the patch geometry.

#### Example: Single Smooth Curve

```c
curve_seq_t curve_seq;
curve_seq_init(&curve_seq);

curve_t curve_1;
curve_seq_add_curve(
    &curve_seq, &curve_1,
    (scalar_func_t) l_1,        (scalar_func_t) l_2,
    (scalar_func_t) l_1_prime,  (scalar_func_t) l_2_prime,
    (scalar_func_t) l_1_dprime, (scalar_func_t) l_2_dprime,
    0,     /* n:          auto-compute from arc length */
    0.1,   /* frac_n_C_0: corner patch at theta=0 spans 10% of n */
    0.1,   /* frac_n_C_1: corner patch at theta=1 spans 10% of n */
    0.6,   /* frac_n_S_0: S-patch overlaps 60% into the theta=0 corner patch */
    0.6,   /* frac_n_S_1: S-patch overlaps 60% into the theta=1 corner patch */
    h_norm);
```

For a multi-curve domain, call `curve_seq_add_curve` once for each curve in counter-clockwise order. `curve_seq.plot_geometry(d)` in the [MATLAB reference implementation](https://github.com/ayang923/2dfc-matlab) can be used to visualize the patch decomposition before running the full C algorithm.

---

### 3. Call `FC2D`

Two variants of the main entry point are provided:

- **`FC2D`** — uses stack (VLA) allocation for intermediate arrays. Suitable for small to medium problems.
- **`FC2D_heap`** — identical algorithm but uses `malloc` for intermediate arrays. Use this when the default stack size is insufficient (the program crashes with a segfault on large inputs).

```c
/* Stack allocation (small/medium problems): */
FC2D(f, h, curve_seq, eps_xi_eta, eps_xy, d, C, n_r, A, Q, M, n_x_fft, n_y_fft);

/* Heap allocation (large problems): */
FC2D_heap(f, h, curve_seq, eps_xi_eta, eps_xy, d, C, n_r, A, Q, M, n_x_fft, n_y_fft);
```

Pass `-1` for `n_x_fft` and `n_y_fft` to use the natural Cartesian grid dimensions, or provide explicit sizes to pad the FFT to a more FFT-friendly grid (e.g., a highly composite number):

```c
MKL_INT n_x_padded = 2916;
MKL_INT n_y_padded = 3888;
FC2D(f, h, curve_seq, 1e-13, 1e-13, d, C, n_r, A, Q, M, n_x_padded, n_y_padded);
```

`FC2D` prints the absolute max error, relative max error, and relative $L^2$ error to stdout as a quick accuracy check.

---

## Examples

The `examples/` directory contains the code used for the examples in the paper.

To build and run an example, set `TARGETS` in the `Makefile` to the example name (without `.c`) and run `make`:

```bash
# Edit Makefile: TARGETS = boomerang_2DFC
make
./bin/boomerang_2DFC
```

---

## Troubleshooting

**Error is large or doesn't converge:**
- This implementation has no built-in geometry visualization. Use the MATLAB reference implementation ([ayang923/2dfc-matlab](https://github.com/ayang923/2dfc-matlab)) to prototype the geometry: call `curve_seq.plot_geometry(d)` there to visually inspect patch placement before porting parameters to C. In general, it is easier to tune patch parameters (`frac_n_C_0/1`, `frac_n_S_0/1`, `h_norm`) on a coarse mesh in MATLAB first, then transfer the validated parameters here for higher-resolution runs.
- Adjust `frac_n_C_0/1` and `frac_n_S_0/1` if patches overlap incorrectly or leave gaps near corners.
- Try increasing `h_norm` or widening the patch overlap regions to make the blending-to-zero strip larger.

**Newton solver non-convergence warnings:**
- These usually arise at extreme patch boundaries; a small number are tolerable
- This can happen if patch decomposition is inconsistent or polygonal approximation of global boundary is not refined enough
- Loosen `eps_xi_eta` / `eps_xy` if the issue is widespread

**Stack overflow / segfault on large problems:**
- Increase stack limit
- Switch from `FC2D` to `FC2D_heap`, which heap-allocates all large intermediate arrays.

**Need FC matrices for new parameters:**
- Use `generate_bdry_continuations` in the [2dfc-matlab](https://github.com/ayang923/2dfc-matlab) repository (requires MATLAB with the Symbolic Math Toolbox; takes several minutes for large `d`).
- Copy the resulting `.txt` files into `fc_data/`.
