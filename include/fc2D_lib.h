#ifndef __FC2D_LIB__
#define __FC2D_LIB__

#include <stdlib.h>

#include "num_linalg_lib.h"
#include "curve_seq_lib.h"

/**
 * @brief Runs the full 2D Fourier Continuation pipeline using stack-allocated arrays.
 *
 * Performs all stages of the 2DFC algorithm:
 *   1. Patch construction (S- and C-patches with partition-of-unity blending).
 *   2. 1D Fourier Continuation along each patch's normal direction.
 *   3. Cartesian mesh initialization and interior masking.
 *   4. Patch-to-mesh interpolation.
 *   5. Interior fill with exact values.
 *   6. 2D FFT and error computation.
 *
 * All intermediate arrays are allocated on the stack; use FC2D_heap for
 * large problems that exceed the default stack size.
 *
 * @param f           Target function f(x, y) to approximate.
 * @param h           Uniform grid spacing (used for both normal and Cartesian grids).
 * @param curve_seq   Circular sequence of boundary curves describing the domain.
 * @param eps_xi_eta  Newton-method convergence tolerance in parameter space.
 * @param eps_xy      Newton-method convergence tolerance in physical space.
 * @param d           1D-BTZ matching mesh size.
 * @param C           Number of continuation points.
 * @param n_r         Refinement factor.
 * @param A           Precomputed FC continuation matrix A (from read_fc_matrix).
 * @param Q           Precomputed FC gram-blend matrix Q (from read_fc_matrix).
 * @param M           Barycentric interpolation stencil width (typically d + 3).
 * @param n_x_fft     FFT grid width  (-1 to use n_x of the Cartesian mesh).
 * @param n_y_fft     FFT grid height (-1 to use n_y of the Cartesian mesh).
 */
void FC2D(scalar_func_2D_t f, double h, curve_seq_t curve_seq, double eps_xi_eta, double eps_xy, MKL_INT d, MKL_INT C, MKL_INT n_r, rd_mat_t A, rd_mat_t Q, MKL_INT M, MKL_INT n_x_fft, MKL_INT n_y_fft);

/**
 * @brief Same as FC2D but uses heap allocation for large intermediate arrays.
 *
 * Identical algorithm to FC2D; intermediate arrays whose sizes depend on the
 * domain discretization are malloc'd rather than declared as VLAs.  All
 * allocations are freed before returning.
 *
 * @param f           Target function f(x, y) to approximate.
 * @param h           Uniform grid spacing.
 * @param curve_seq   Circular sequence of boundary curves.
 * @param eps_xi_eta  Newton-method convergence tolerance in parameter space.
 * @param eps_xy      Newton-method convergence tolerance in physical space.
 * @param d           1D-BTZ matching mesh size.
 * @param C           Number of continuation points.
 * @param n_r         Refinement factor.
 * @param A           Precomputed FC continuation matrix A.
 * @param Q           Precomputed FC gram-blend matrix Q.
 * @param M           Barycentric interpolation stencil width.
 * @param n_x_fft     FFT grid width  (-1 to use n_x of the Cartesian mesh).
 * @param n_y_fft     FFT grid height (-1 to use n_y of the Cartesian mesh).
 */
void FC2D_heap(scalar_func_2D_t f, double h, curve_seq_t curve_seq, double eps_xi_eta, double eps_xy, MKL_INT d, MKL_INT C, MKL_INT n_r, rd_mat_t A, rd_mat_t Q, MKL_INT M, MKL_INT n_x_fft, MKL_INT n_y_fft);

#endif
