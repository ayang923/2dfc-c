#ifndef __S_PATCH_LIB_H__
#define __S_PATCH_LIB_H__

#include "q_patch_lib.h"

/**
 * @brief Function pointer for a generalized M_p that also takes the ratio H.
 *
 * Used by smooth (S) patches, where the map depends on H (the ratio of
 * real-space normal mesh size to parameter-space eta mesh size) in addition
 * to (ξ, η).
 *
 * @param xi           Matrix of ξ values.
 * @param eta          Matrix of η values (same shape as xi).
 * @param H            Ratio of real-space normal mesh size to parameter-space eta mesh size.
 * @param x            Output x-coordinates.
 * @param y            Output y-coordinates.
 * @param extra_param  Optional user-provided parameter struct.
 */
typedef void(*M_p_general_handle_t) (rd_mat_t xi, rd_mat_t eta, double H, rd_mat_t *x, rd_mat_t *y, void* extra_param);

/**
 * @brief Function pointer for a generalized Jacobian that also takes H.
 *
 * @param v            2-element column vector [ξ; η].
 * @param H            Ratio of real-space normal mesh size to parameter-space eta mesh size.
 * @param J_vals       Output 2×2 Jacobian.
 * @param extra_param  Optional user-provided parameter struct.
 */
typedef void (*J_general_handle_t) (rd_mat_t v, double H, rd_mat_t *J_vals, void* extra_param);

/**
 * @brief Bundles a generalized Jacobian function pointer with its extra parameters.
 */
typedef struct J_general {
    J_general_handle_t J_general_handle;
    void* extra_param;
} J_general_t;

/**
 * @brief Bundles a generalized M_p function pointer with its extra parameters.
 */
typedef struct M_p_general {
    M_p_general_handle_t M_p_general_handle;
    void* extra_param;
} M_p_general_t;

/**
 * @brief Smooth boundary patch (S-patch).
 *
 * An S-patch covers the smooth portion of a boundary curve.  It wraps a
 * q_patch_t (accessible as Q) and stores the generalized map M_p_general
 * together with the normal-direction step h and the ratio H.
 *
 * @param Q             Underlying q_patch (ξ ∈ [0,1], η ∈ [0, (d-1)*h]).
 * @param M_p_general   Generalized parametric map with normal parameter H.
 * @param J_general     Generalized Jacobian with normal parameter H.
 * @param h             Grid spacing in the normal direction.
 * @param H             Ratio of real-space normal mesh size to parameter-space eta mesh size.
 */
typedef struct s_patch {
    q_patch_t Q;
    M_p_general_t M_p_general;
    J_general_t J_general;
    double h;
    double H;
} s_patch_t;

/**
 * @brief Initializes a pre-allocated s_patch_t.
 *
 * Sets up the underlying q_patch with ξ ∈ [0, 1] and η ∈ [0, (d-1)*h],
 * wiring the generalized M_p and J through thin wrapper functions.
 *
 * @param s_patch       Pointer to the s_patch to initialize.
 * @param M_p_general   Generalized parametric map.
 * @param J_general     Generalized Jacobian.
 * @param h             Normal-direction grid spacing.
 * @param eps_xi_eta    Convergence tolerance in (ξ, η) space.
 * @param eps_xy        Convergence tolerance in (x, y) space.
 * @param n_xi          Number of grid points in the ξ (tangential) direction.
 * @param d             1D-BTZ matching mesh size.
 * @param f_XY          Pre-allocated (d × n_xi) matrix for sampled function values.
 */
void s_patch_init(s_patch_t *s_patch, M_p_general_t M_p_general, J_general_t J_general, double h, double eps_xi_eta, double eps_xy, MKL_INT n_xi, MKL_INT d, rd_mat_t *f_XY);

/**
 * @brief Returns the number of elements in the extension-mesh for an S-patch.
 *
 * The FC grid has (C*n_r + 1) rows (the continuation) and n_xi columns.
 *
 * @param s_patch  The S-patch.
 * @param C        Number of continuation points.
 * @param n_r      Refinement factor.
 * @return         (C*n_r + 1) * n_xi.
 */
MKL_INT s_patch_FC_num_el(s_patch_t *s_patch, MKL_INT C, MKL_INT n_r);

/**
 * @brief Computes the 1D Fourier Continuation for each column of an S-patch.
 *
 * For each ξ-column of the patch, applies fcont_gram_blend_S along the η
 * direction to produce a C-periodic extension of length C*n_r+1.  The result
 * is stored in f_FC and a new q_patch_t (s_patch_FC) is initialized to cover
 * the extended η domain.
 *
 * @param s_patch      The smooth patch.
 * @param C            Number of continuation points.
 * @param n_r          Refinement factor.
 * @param d            1D-BTZ matching mesh size.
 * @param A            FC continuation matrix A (C*n_r × d).
 * @param Q            FC orthogonalization matrix Q (d × d).
 * @param s_patch_FC   Output: initialized q_patch covering the extended domain.
 * @param f_FC         Output: matrix wrapper pointing into data_stack.
 * @param data_stack   Pre-allocated buffer of length s_patch_FC_num_el(s_patch, C, n_r).
 * @return             Number of elements written to data_stack.
 */
MKL_INT s_patch_FC(s_patch_t *s_patch, MKL_INT C, MKL_INT n_r, MKL_INT d, rd_mat_t A, rd_mat_t Q, q_patch_t* s_patch_FC, rd_mat_t *f_FC, double *data_stack);

#endif
