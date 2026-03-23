#ifndef __R_CARTESIAN_MESH_LIB__
#define __R_CARTESIAN_MESH_LIB__

#include "num_linalg_lib.h"
#include <mkl.h>
#include <string.h>

#include "q_patch_lib.h"

/**
 * @brief Tests whether each point of a regular meshgrid lies inside a polygon.
 *
 * Uses a ray-casting algorithm along horizontal scanlines.  The polygon is
 * given as a closed boundary polyline (boundary_X, boundary_Y).
 *
 * @param R_X        (n_y × n_x) meshgrid of x-coordinates (column-major).
 * @param R_Y        (n_y × n_x) meshgrid of y-coordinates.
 * @param boundary_X Column vector of boundary x-coordinates (closed: first == last).
 * @param boundary_Y Column vector of boundary y-coordinates.
 * @param in_msk     Output integer matrix (same shape as R_X); 1 = interior, 0 = exterior.
 * @return           Number of interior mesh points.
 */
MKL_INT inpolygon_mesh(rd_mat_t R_X, rd_mat_t R_Y, rd_mat_t boundary_X, rd_mat_t boundary_Y, ri_mat_t *in_msk);

/**
 * @brief Regular Cartesian output mesh covering the bounding box of the domain.
 *
 * @param x_start, x_end  Requested x extent (x_end is rounded up to a grid line).
 * @param y_start, y_end  Requested y extent (y_end is rounded up to a grid line).
 * @param h               Uniform grid spacing.
 * @param n_x, n_y        Actual number of grid lines in x and y after rounding.
 * @param R_X             Pointer to (n_y × n_x) meshgrid of x-coordinates.
 * @param R_Y             Pointer to (n_y × n_x) meshgrid of y-coordinates.
 * @param in_interior     Pointer to (n_y × n_x) interior mask.
 * @param f_R             Pointer to (n_y × n_x) matrix accumulating FC values.
 */
typedef struct r_cartesian_mesh_obj {
    double x_start;
    double x_end;
    double y_start;
    double y_end;
    double h;
    MKL_INT n_x;
    MKL_INT n_y;

    rd_mat_t *R_X;
    rd_mat_t *R_Y;

    ri_mat_t *in_interior;
    rd_mat_t *f_R;
} r_cartesian_mesh_obj_t;

/**
 * @brief Returns the total number of grid points for a given bounding box and spacing.
 *
 * @param x_start, x_end  x-extent of the bounding box.
 * @param y_start, y_end  y-extent of the bounding box.
 * @param h               Grid spacing.
 * @return                n_x(h) * n_y(h).
 */
MKL_INT r_cartesian_n_total(double x_start, double x_end, double y_start, double y_end, double h);

void r_cartesian_mesh_init(r_cartesian_mesh_obj_t *r_cartesian_mesh_obj, double x_start, double x_end, double y_start, double y_end, double h, rd_mat_t boundary_X, rd_mat_t boundary_Y, rd_mat_t *R_X, rd_mat_t *R_Y, ri_mat_t *in_interior, rd_mat_t *f_R);

/**
 * @brief Interpolates a single FC patch onto the Cartesian mesh (stack version).
 *
 * Finds all Cartesian mesh points inside the patch boundary, inverts M_p for
 * each via Newton's method, and evaluates f using tensor-product barycentric
 * interpolation.  Adds the result into f_R (does not overwrite; interior
 * points that were already set by inpolygon_mesh are excluded).
 *
 * @param r_cartesian_mesh_obj  The output Cartesian mesh.
 * @param q_patch               The FC patch to interpolate.
 * @param M                     Barycentric stencil width.
 */
void r_cartesian_mesh_interpolate_patch(r_cartesian_mesh_obj_t *r_cartesian_mesh_obj, q_patch_t *q_patch, MKL_INT M);

/**
 * @brief Same as r_cartesian_mesh_interpolate_patch but uses heap allocation.
 *
 * @param r_cartesian_mesh_obj  The output Cartesian mesh.
 * @param q_patch               The FC patch to interpolate.
 * @param M                     Barycentric stencil width.
 */
void r_cartesian_mesh_interpolate_patch_heap(r_cartesian_mesh_obj_t *r_cartesian_mesh_obj, q_patch_t *q_patch, MKL_INT M);

/**
 * @brief Fills interior mesh points with exact function values.
 *
 * For every grid point marked interior by in_interior, overwrites f_R with
 * the exact value f(x, y). 
 *
 * @param r_cartesian_mesh_obj  The Cartesian mesh (f_R is modified in place).
 * @param f                     The exact function f(x, y).
 */
void r_cartesian_mesh_fill_interior(r_cartesian_mesh_obj_t *r_cartesian_mesh_obj, scalar_func_2D_t f);

/**
 * @brief Computes the 2DFC approximation error on a refined interior mesh (stack version).
 *
 * Performs a 2D FFT of f_R (zero-padded to n_x_fft × n_y_fft), then
 * evaluates the inverse FFT on a rho_err-times-finer mesh inside the domain
 * and reports the L∞ and relative L2 errors against the exact function.
 *
 * @param r_cartesian_mesh_obj  The Cartesian mesh containing the FC values.
 * @param f                     Exact function for comparison.
 * @param rho_err               Refinement factor for the error evaluation mesh.
 * @param boundary_X            Closed boundary x-coordinates for the interior mask.
 * @param boundary_Y            Closed boundary y-coordinates.
 * @param n_x_fft               FFT grid width.
 * @param n_y_fft               FFT grid height.
 * @return                      L∞ absolute error over the interior.
 */
double r_cartesian_mesh_compute_fc_error(r_cartesian_mesh_obj_t *r_cartesian_mesh_obj, scalar_func_2D_t f, MKL_INT rho_err, rd_mat_t boundary_X, rd_mat_t boundary_Y, MKL_INT n_x_fft, MKL_INT n_y_fft);

/**
 * @brief Same as r_cartesian_mesh_compute_fc_error but uses heap allocation.
 *
 * @param r_cartesian_mesh_obj  The Cartesian mesh containing the FC values.
 * @param f                     Exact function for comparison.
 * @param rho_err               Refinement factor for the error evaluation mesh.
 * @param boundary_X            Closed boundary x-coordinates.
 * @param boundary_Y            Closed boundary y-coordinates.
 * @param n_x_fft               FFT grid width.
 * @param n_y_fft               FFT grid height.
 * @return                      L∞ absolute error over the interior.
 */
double r_cartesian_mesh_compute_fc_error_heap(r_cartesian_mesh_obj_t *r_cartesian_mesh_obj, scalar_func_2D_t f, MKL_INT rho_err, rd_mat_t boundary_X, rd_mat_t boundary_Y, MKL_INT n_x_fft, MKL_INT n_y_fft);

#endif
