#ifndef __Q_PATCH_LIB_H__
#define __Q_PATCH_LIB_H__

#include <stdlib.h>
#include "num_linalg_lib.h"
#include <stdbool.h>

/**
 * @brief Function pointer type for the parametric map M_p: (ξ, η) → (x, y).
 *
 * @param xi           Matrix of ξ (xi) parameter values.
 * @param eta          Matrix of η (eta) parameter values (same shape as xi).
 * @param x            Output matrix of x-coordinates (same shape as xi).
 * @param y            Output matrix of y-coordinates (same shape as xi).
 * @param extra_param  Optional user-provided parameter struct (may be NULL).
 */
typedef void (*M_p_handle_t) (rd_mat_t xi, rd_mat_t eta, rd_mat_t *x, rd_mat_t *y, void* extra_param);

/**
 * @brief Bundles an M_p function pointer with its extra parameters.
 *
 * @param M_p_handle   The M_p function pointer.
 * @param extra_param  User-provided parameter struct forwarded to M_p_handle.
 */
typedef struct M_p {
    M_p_handle_t M_p_handle;
    void* extra_param;
} M_p_t;

/**
 * @brief Function pointer type for the Jacobian J of M_p at a single point.
 *
 * @param v            2-element column vector [ξ; η].
 * @param J_vals       Output 2×2 Jacobian matrix (column-major).
 * @param extra_param  Optional user-provided parameter struct (may be NULL).
 */
typedef void (*J_handle_t) (rd_mat_t v, rd_mat_t *J_vals, void* extra_param);

/**
 * @brief Bundles a Jacobian function pointer with its extra parameters.
 *
 * @param J_handle     The Jacobian function pointer.
 * @param extra_param  User-provided parameter struct forwarded to J_handle.
 */
typedef struct J {
    J_handle_t J_handle;
    void* extra_param;
} J_t;

/**
 * @brief Function pointer type for the 1D window function w(θ).
 *
 * @param theta       Input values (column vector).
 * @param w_1D_vals   Output values (same shape as theta).
 */
typedef void (*w_1D_t) (rd_mat_t theta, rd_mat_t *w_1D_vals);

/**
 * @brief Rectangular parametric patch: the fundamental patch construction unit.
 *
 * A q_patch maps a rectangular parameter domain [xi_start, xi_end] ×
 * [eta_start, eta_end] to physical (x, y) space via M_p.  The function
 * values f(x, y) sampled on the resulting grid are stored in f_XY.
 *
 * @param M_p           Parametrization M_p: (ξ, η) → (x, y).
 * @param J             Jacobian of M_p.
 * @param n_xi          Number of grid points in the ξ direction.
 * @param n_eta         Number of grid points in the η direction.
 * @param xi_start      Start of the ξ parameter interval.
 * @param xi_end        End of the ξ parameter interval.
 * @param eta_start     Start of the η parameter interval.
 * @param eta_end       End of the η parameter interval.
 * @param h_xi          Spacing between ξ grid points.
 * @param h_eta         Spacing between η grid points.
 * @param f_XY          (n_eta × n_xi) matrix of sampled function values.
 * @param w_1D          1D window function used for partition-of-unity blending.
 * @param eps_xi_eta    Convergence tolerance in parameter space (ξ, η).
 * @param eps_xy        Convergence tolerance in physical space (x, y).
 * @param x_min/x_max   Bounding box of the physical-space image of the patch.
 * @param y_min/y_max
 */
typedef struct q_patch {
    M_p_t M_p;
    J_t J;

    MKL_INT n_xi;
    MKL_INT n_eta;
    double xi_start;
    double xi_end;
    double eta_start;
    double eta_end;

    double h_xi;
    double h_eta;

    rd_mat_t *f_XY;
    w_1D_t w_1D;

    double eps_xi_eta;
    double eps_xy;

    double x_min;
    double y_min;
    double x_max;
    double y_max;
} q_patch_t;

/**
 * @brief Initializes a pre-allocated q_patch_t.
 *
 * Sets all fields, computes grid spacings h_xi / h_eta, sets f_XY shape,
 * and computes the (x, y) bounding box of the patch image.
 *
 * @param q_patch    Pointer to the patch to initialize.
 * @param M_p        Parametric map.
 * @param J          Jacobian of M_p.
 * @param eps_xi_eta Convergence tolerance in (ξ, η) space.
 * @param eps_xy     Convergence tolerance in (x, y) space.
 * @param n_xi       Grid points in ξ direction.
 * @param n_eta      Grid points in η direction.
 * @param xi_start   Start of ξ interval.
 * @param xi_end     End of ξ interval.
 * @param eta_start  Start of η interval.
 * @param eta_end    End of η interval.
 * @param f_XY       Pre-allocated matrix that will hold sampled function values.
 */
void q_patch_init(q_patch_t *q_patch, M_p_t M_p, J_t J, double eps_xi_eta, double eps_xy, MKL_INT n_xi, MKL_INT n_eta, double xi_start, double xi_end, double eta_start, double eta_end, rd_mat_t *f_XY);

/**
 * @brief Returns the total number of grid points (n_xi * n_eta).
 *
 * @param q_patch  The patch.
 * @return         n_xi * n_eta.
 */
MKL_INT q_patch_grid_num_el(q_patch_t *q_patch);

/**
 * @brief Evaluates M_p at the given (ξ, η) arrays and writes (x, y) outputs.
 *
 * @param q_patch  The patch whose M_p is evaluated.
 * @param xi       Input ξ values.
 * @param eta      Input η values (same shape as xi).
 * @param x        Output x-coordinates (shaped to match xi).
 * @param y        Output y-coordinates (shaped to match xi).
 */
void q_patch_evaluate_M_p(q_patch_t *q_patch, rd_mat_t xi, rd_mat_t eta, rd_mat_t *x, rd_mat_t *y);

/**
 * @brief Evaluates the Jacobian of M_p at a single point v = [ξ; η].
 *
 * @param q_patch  The patch.
 * @param v        2-element column vector [ξ; η].
 * @param J_vals   Output 2×2 Jacobian matrix.
 */
void q_patch_evaluate_J(q_patch_t *q_patch, rd_mat_t v, rd_mat_t *J_vals);

/**
 * @brief Computes the 1D ξ grid (linspace from xi_start to xi_end, n_xi points).
 *
 * @param q_patch        The patch.
 * @param xi_mesh_vals   Output column vector of length n_xi.
 */
void q_patch_xi_mesh(q_patch_t *q_patch, rd_mat_t *xi_mesh_vals);

/**
 * @brief Computes the 1D η grid (linspace from eta_start to eta_end, n_eta points).
 *
 * @param q_patch         The patch.
 * @param eta_mesh_vals   Output column vector of length n_eta.
 */
void q_patch_eta_mesh(q_patch_t *q_patch, rd_mat_t *eta_mesh_vals);

/**
 * @brief Computes the 2D meshgrid of (ξ, η) parameter values over the patch.
 *
 * @param q_patch    The patch.
 * @param XI_vals    Output (n_eta × n_xi) matrix of ξ values.
 * @param ETA_vals   Output (n_eta × n_xi) matrix of η values.
 */
void q_patch_xi_eta_mesh(q_patch_t *q_patch, rd_mat_t *XI_vals, rd_mat_t *ETA_vals);

/**
 * @brief Converts a (ξ, η) meshgrid to physical (x, y) coordinates via M_p.
 *
 * @param q_patch  The patch.
 * @param XI       Matrix of ξ values.
 * @param ETA      Matrix of η values (same shape as XI).
 * @param X_vals   Output x-coordinates (same shape as XI).
 * @param Y_vals   Output y-coordinates (same shape as XI).
 */
void q_patch_convert_to_XY(q_patch_t *q_patch, rd_mat_t XI, rd_mat_t ETA, rd_mat_t *X_vals, rd_mat_t *Y_vals);

/**
 * @brief Computes the physical (x, y) meshgrid for all grid points of the patch.
 *
 * @param q_patch  The patch.
 * @param X_vals   Output (n_eta × n_xi) matrix of x-coordinates.
 * @param Y_vals   Output (n_eta × n_xi) matrix of y-coordinates.
 */
void q_patch_xy_mesh(q_patch_t *q_patch, rd_mat_t *X_vals, rd_mat_t *Y_vals);

/**
 * @brief Evaluates f at all (x, y) grid points and stores the results in f_XY.
 *
 * @param q_patch  The patch (f_XY is written in place).
 * @param f        Scalar function f(x, y) to sample.
 */
void q_patch_evaluate_f(q_patch_t *q_patch, scalar_func_2D_t f);

/**
 * @brief Returns the number of points in the boundary mesh of the patch.
 *
 * The boundary traversal visits all four sides once (2*n_xi + 2*n_eta + 1).
 *
 * @param q_patch  The patch.
 * @return         Total number of boundary mesh points.
 */
MKL_INT q_patch_boundary_mesh_num_el(q_patch_t *q_patch);

/**
 * @brief Computes the (ξ, η) coordinates of the patch boundary traversed CCW.
 *
 * @param q_patch              The patch.
 * @param pad_boundary         If true, expand the boundary by one grid spacing.
 * @param boundary_mesh_xi     Output ξ coordinates of boundary points.
 * @param boundary_mesh_eta    Output η coordinates of boundary points.
 */
void q_patch_boundary_mesh(q_patch_t *q_patch, bool pad_boundary, rd_mat_t *boundary_mesh_xi, rd_mat_t *boundary_mesh_eta);

/**
 * @brief Computes the physical (x, y) coordinates of the patch boundary.
 *
 * @param q_patch           The patch.
 * @param pad_boundary      If true, expand the boundary by one grid spacing.
 * @param boundary_mesh_x   Output x coordinates of boundary points.
 * @param boundary_mesh_y   Output y coordinates of boundary points.
 */
void q_patch_boundary_mesh_xy(q_patch_t *q_patch, bool pad_boundary, rd_mat_t *boundary_mesh_x, rd_mat_t *boundary_mesh_y);

/**
 * @brief Applies the partition-of-unity window normalization on the right ξ edge.
 *
 * In the overlap region between main_patch and window_patch, scales f_XY of
 * main_patch by w_main / (w_main + w_window) so that the two patches blend
 * smoothly at their shared boundary.
 *
 * @param main_patch    Patch whose f_XY is modified.
 * @param window_patch  Neighboring patch that provides the complementary weight.
 */
void q_patch_apply_w_normalization_xi_right(q_patch_t *main_patch, q_patch_t *window_patch);

/** @brief Same as q_patch_apply_w_normalization_xi_right but on the left ξ edge. */
void q_patch_apply_w_normalization_xi_left(q_patch_t *main_patch, q_patch_t *window_patch);

/** @brief Same as q_patch_apply_w_normalization_xi_right but on the upper η edge. */
void q_patch_apply_w_normalization_eta_up(q_patch_t *main_patch, q_patch_t *window_patch);

/** @brief Same as q_patch_apply_w_normalization_xi_right but on the lower η edge. */
void q_patch_apply_w_normalization_eta_down(q_patch_t *main_patch, q_patch_t *window_patch);

/**
 * @brief Return type for q_patch_inverse_M_p.
 *
 * @param xi         Computed ξ parameter (valid only if converged != 0).
 * @param eta        Computed η parameter (valid only if converged != 0).
 * @param converged  Non-zero if Newton's method converged to an in-patch point.
 */
typedef struct inverse_M_p_return_type {
    double xi;
    double eta;
    int converged;
} inverse_M_p_return_type_t;

/**
 * @brief Inverts M_p at a physical point (x, y) using Newton's method.
 *
 * Tries each supplied initial guess in order and returns the first (ξ, η) that
 * both converges and lies inside the patch.  If initial_guesses_xi/eta are NULL,
 * a default set of 20 guesses distributed over the parameter domain boundary is used.
 *
 * @param q_patch               The patch.
 * @param x                     Target x coordinate.
 * @param y                     Target y coordinate.
 * @param initial_guesses_xi    Column vector of initial ξ guesses (or NULL).
 * @param initial_guesses_eta   Column vector of initial η guesses (or NULL).
 * @return                      Struct with (xi, eta, converged).
 */
inverse_M_p_return_type_t q_patch_inverse_M_p(q_patch_t *q_patch, double x, double y, rd_mat_t* initial_guesses_xi, rd_mat_t* initial_guesses_eta);

/**
 * @brief Return type for q_patch_locally_compute.
 *
 * @param f_xy      Interpolated function value (NAN if not in patch).
 * @param in_range  1 if the (ξ, η) point lies inside the patch, 0 otherwise.
 */
typedef struct locally_compute_return_type {
    double f_xy;
    int in_range;
} locally_compute_return_type_t;

/**
 * @brief Interpolates f at (ξ, η) using a local M×M barycentric stencil.
 *
 * Selects the M nearest grid nodes in each parameter direction and applies
 * tensor-product barycentric Lagrange interpolation.  Returns NAN if the
 * point is outside the patch.
 *
 * @param q_patch  The patch (f_XY must already be filled).
 * @param xi       ξ parameter value.
 * @param eta      η parameter value.
 * @param M        Stencil width (use an even number for centered stencils).
 * @return         Struct with (f_xy, in_range).
 */
locally_compute_return_type_t q_patch_locally_compute(q_patch_t *q_patch, double xi, double eta, int M);

#endif
