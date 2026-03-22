#ifndef __C_PATCH_LIB_H__
#define __C_PATCH_LIB_H__

#include <mkl.h>
#include <stdlib.h>

#include "num_linalg_lib.h"
#include "q_patch_lib.h"
#include "s_patch_lib.h"

/**
 * @brief Discriminates between the two corner-patch geometric cases.
 *
 * C2: convex corner
 * C1: concave corner
 */
typedef enum c_patch_type {
    C1,
    C2,
} c_patch_type_t;

/**
 * @brief Corner patch: covers the neighborhood of a corner where two curves meet.
 *
 * For ease of implementation, the corner patch is decomposed into two rectangular sub-patches.
 *
 * @param L             Long/tall sub-patch
 * @param W             Wide/short sub-patch
 * @param c_patch_type  Whether this is a C1 (concave) or C2 (convex) corner.
 */
typedef struct c_patch {
    q_patch_t L;
    q_patch_t W;
    c_patch_type_t c_patch_type;
} c_patch_t;

/**
 * @brief Initializes a C1 (concave) corner patch.
 *
 * @param c_patch     Pointer to the corner patch to initialize.
 * @param M_p         Parametric map for the corner region.
 * @param J           Jacobian of M_p.
 * @param eps_xi_eta  Convergence tolerance in (ξ, η) space.
 * @param eps_xy      Convergence tolerance in (x, y) space.
 * @param n_xi        Number of grid points in the ξ direction.
 * @param n_eta       Number of grid points in the η direction.
 * @param d           1D-BTZ matching mesh size.
 * @param f_L         Pre-allocated data matrix for the L sub-patch.
 * @param f_W         Pre-allocated data matrix for the W sub-patch.
 */
void c1_patch_init(c_patch_t *c_patch, M_p_t M_p, J_t J, double eps_xi_eta, double eps_xy, MKL_INT n_xi, MKL_INT n_eta, MKL_INT d, rd_mat_t *f_L, rd_mat_t *f_W);

/**
 * @brief Initializes a C2 (convex) corner patch.
 *
 * @param c_patch     Pointer to the corner patch to initialize.
 * @param M_p         Parametric map for the corner region.
 * @param J           Jacobian of M_p.
 * @param eps_xi_eta  Convergence tolerance in (ξ, η) space.
 * @param eps_xy      Convergence tolerance in (x, y) space.
 * @param n_xi        Number of grid points in the ξ direction.
 * @param n_eta       Number of grid points in the η direction.
 * @param d           1D-BTZ matching mesh size.
 * @param f_L         Pre-allocated data matrix for the L sub-patch.
 * @param f_W         Pre-allocated data matrix for the W sub-patch.
 */
void c2_patch_init(c_patch_t *c_patch, M_p_t M_p, J_t J, double eps_xi_eta, double eps_xy, MKL_INT n_xi, MKL_INT n_eta, MKL_INT d, rd_mat_t *f_L, rd_mat_t *f_W);

/**
 * @brief Applies the partition-of-unity window to the W sub-patch of a C1 corner.
 *
 * @param c_patch        The C1 corner patch (W is modified).
 * @param window_patch_W The neighboring S-patch that provides the blending weight.
 */
void c1_patch_apply_w_W(c_patch_t *c_patch, s_patch_t *window_patch_W);

/**
 * @brief Applies the partition-of-unity window to the L sub-patch of a C1 corner.
 *
 * @param c_patch        The C1 corner patch (L is modified).
 * @param window_patch_L The neighboring S-patch that provides the blending weight.
 */
void c1_patch_apply_w_L(c_patch_t *c_patch, s_patch_t *window_patch_L);

/** @brief Applies the partition-of-unity window to the W sub-patch of a C2 corner. */
void c2_patch_apply_w_W(c_patch_t *c_patch, s_patch_t *window_patch_W);

/** @brief Applies the partition-of-unity window to the L sub-patch of a C2 corner. */
void c2_patch_apply_w_L(c_patch_t *c_patch, s_patch_t *window_patch_L);

/** @brief Returns the number of elements in the FC-extended W data for a C1 patch. */
MKL_INT c1_patch_FC_W_num_el(c_patch_t *c_patch, MKL_INT C, MKL_INT n_r);

/** @brief Returns the number of elements in the FC-extended W data for a C2 patch. */
MKL_INT c2_patch_FC_W_num_el(c_patch_t *c_patch, MKL_INT C, MKL_INT n_r);

/** @brief Returns the number of elements in the FC-extended L data for a C1 patch. */
MKL_INT c1_patch_FC_L_num_el(c_patch_t *c_patch, MKL_INT C, MKL_INT n_r, MKL_INT d);

/** @brief Returns the number of elements in the FC-extended L data for a C2 patch. */
MKL_INT c2_patch_FC_L_num_el(c_patch_t *c_patch, MKL_INT C, MKL_INT n_r, MKL_INT d);

/**
 * @brief Returns the number of elements in the corner region of the 2D-BTZ extension.
 *
 * The corner sub-patch is a (C*n_r+1) × (C*n_r+1) square.
 *
 * @param C    Number of continuation points.
 * @param n_r  Refinement factor.
 * @return     (C*n_r + 1)^2.
 */
MKL_INT c_patch_FC_corner_num_el(MKL_INT C, MKL_INT n_r);

/**
 * @brief Computes Fourier Continuations for all three sub-regions of a C1 corner.
 *
 * Produces three FC patches (L, W, corner) and returns the total number of
 * elements written to data_stack.
 *
 * @param c_patch             The C1 corner patch.
 * @param C                   Number of continuation points.
 * @param n_r                 Refinement factor.
 * @param d                   1D-BTZ matching mesh size.
 * @param A                   FC continuation matrix A.
 * @param Q                   FC orthogonalization matrix Q.
 * @param M                   Barycentric interpolation stencil width.
 * @param c1_fcont_patch_L    Output: FC patch for the L sub-region.
 * @param c1_fcont_patch_W    Output: FC patch for the W sub-region.
 * @param c1_fcont_patch_corner Output: FC patch for the corner sub-region.
 * @param f_L                 Output: matrix wrapper for L FC data.
 * @param f_W                 Output: matrix wrapper for W FC data.
 * @param f_corner            Output: matrix wrapper for corner FC data.
 * @param data_stack          Pre-allocated buffer of sufficient size.
 * @return                    Total number of elements written to data_stack.
 */
MKL_INT c1_patch_FC(c_patch_t *c_patch, MKL_INT C, MKL_INT n_r, MKL_INT d, rd_mat_t A, rd_mat_t Q, MKL_INT M, q_patch_t* c1_fcont_patch_L, q_patch_t *c1_fcont_patch_W, q_patch_t *c1_fcont_patch_corner, rd_mat_t *f_L, rd_mat_t *f_W, rd_mat_t *f_corner, double *data_stack);

/**
 * @brief Computes Fourier Continuations for all three sub-regions of a C2 corner.
 *
 * Produces three FC patches (L, W, corner) and returns the total number of
 * elements written to data_stack.
 *
 * @param c_patch             The C2 corner patch.
 * @param C                   Number of continuation points.
 * @param n_r                 Refinement factor.
 * @param d                   1D-BTZ matching mesh size.
 * @param A                   FC continuation matrix A.
 * @param Q                   FC orthogonalization matrix Q.
 * @param c2_fcont_patch_L    Output: FC patch for the L sub-region.
 * @param c2_fcont_patch_W    Output: FC patch for the W sub-region.
 * @param c2_fcont_patch_corner Output: FC patch for the corner sub-region.
 * @param f_L                 Output: matrix wrapper for L FC data.
 * @param f_W                 Output: matrix wrapper for W FC data.
 * @param f_corner            Output: matrix wrapper for corner FC data.
 * @param data_stack          Pre-allocated buffer of sufficient size.
 * @return                    Total number of elements written to data_stack.
 */
MKL_INT c2_patch_FC(c_patch_t *c_patch, MKL_INT C, MKL_INT n_r, MKL_INT d, rd_mat_t A, rd_mat_t Q, q_patch_t* c2_fcont_patch_L, q_patch_t *c2_fcont_patch_W, q_patch_t *c2_fcont_patch_corner, rd_mat_t *f_L, rd_mat_t *f_W, rd_mat_t *f_corner, double *data_stack);

#endif
