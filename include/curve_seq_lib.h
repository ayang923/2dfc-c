#ifndef __CURVE_SEQ_LIB_H__
#define __CURVE_SEQ_LIB_H__

#include <stdlib.h>
#include <mkl.h>

#include "num_linalg_lib.h"
#include "s_patch_lib.h"
#include "c_patch_lib.h"

typedef struct curve curve_t;

/**
 * @brief Parameters for the smooth-patch (S) parametric map M_p_S_general.
 *
 * The map moves a distance η·H in the inward normal direction from
 * the boundary point l(ξ_tilde), where ξ_tilde = xi_diff*ξ + xi_0.
 *
 * @param xi_diff      Scale factor mapping the normalized ξ ∈ [0,1] to arc-length.
 * @param xi_0         Offset mapping normalized ξ to the start of the smooth segment.
 * @param l_1          x-component of the boundary parametrization l(θ).
 * @param l_2          y-component of the boundary parametrization l(θ).
 * @param l_1_prime    First derivative of l_1.
 * @param l_2_prime    First derivative of l_2.
 */
typedef struct M_p_S_general_param {
    double xi_diff;
    double xi_0;

    scalar_func_t l_1;
    scalar_func_t l_2;
    scalar_func_t l_1_prime;
    scalar_func_t l_2_prime;
} M_p_S_general_param_t;

/**
 * @brief Parameters for the corner-patch (C) parametric map M_p_C.
 *
 * The map is defined as:
 *   x(ξ, η) = l_1(ξ_tilde) + next_l_1(η_tilde) − l_1(1)
 *   y(ξ, η) = l_2(ξ_tilde) + next_l_2(η_tilde) − l_2(1)
 * where ξ_tilde = xi_diff*ξ + xi_0 and η_tilde = eta_diff*η + eta_0.
 *
 * @param xi_diff           Scale for the current curve's parameter.
 * @param xi_0              Offset for the current curve's parameter.
 * @param eta_diff          Scale for the next curve's parameter.
 * @param eta_0             Offset for the next curve's parameter.
 * @param l_1               x-component of the current boundary curve.
 * @param l_2               y-component of the current boundary curve.
 * @param next_curve_l_1    x-component of the next boundary curve.
 * @param next_curve_l_2    y-component of the next boundary curve.
 */
typedef struct M_p_C_param {
    double xi_diff;
    double xi_0;
    double eta_diff;
    double eta_0;

    scalar_func_t l_1;
    scalar_func_t l_2;
    scalar_func_t next_curve_l_1;
    scalar_func_t next_curve_l_2;
} M_p_C_param_t;

/**
 * @brief Parameters for the Jacobian of M_p_S_general.
 *
 * Extends M_p_S_general_param_t with the second derivatives needed for the
 * curvature terms in the Jacobian.
 */
typedef struct J_S_general_param {
    double xi_diff;
    double xi_0;

    scalar_func_t l_1;
    scalar_func_t l_2;
    scalar_func_t l_1_prime;
    scalar_func_t l_2_prime;
    scalar_func_t l_1_dprime;
    scalar_func_t l_2_dprime;
} J_S_general_param_t;

/**
 * @brief Parameters for the Jacobian of M_p_C.
 *
 * The Jacobian of M_p_C only requires the first derivatives of both curves
 * (no curvature terms because M_p_C is affine in ξ and η separately).
 */
typedef struct J_C_param {
    double xi_diff;
    double xi_0;
    double eta_diff;
    double eta_0;

    scalar_func_t l_1_prime;
    scalar_func_t l_2_prime;
    scalar_func_t next_curve_l_1_prime;
    scalar_func_t next_curve_l_2_prime;
} J_C_param_t;

/**
 * @brief A single C² boundary curve parametrized on [0, 1].
 *
 * Stores the curve geometry, the discretization counts for the corner (C) and
 * smooth (S) overlap regions, and cached patch-construction parameters that
 * are filled in during patch assembly.
 *
 * @param l_1, l_2             x- and y-components of the parametrization l(θ).
 * @param l_1_prime, l_2_prime First derivatives of l_1 and l_2.
 * @param l_1_dprime, l_2_dprime Second derivatives of l_1 and l_2.
 * @param n                    Total number of grid points along the curve.
 * @param n_C_0                Number of points in the corner overlap at θ = 0.
 * @param n_C_1                Number of points in the corner overlap at θ = 1.
 * @param n_S_0                Number of points in the smooth-patch overlap at θ = 0.
 * @param n_S_1                Number of points in the smooth-patch overlap at θ = 1.
 * @param h_tan                Grid spacing in the tangential direction (1/(n-1)).
 * @param h_norm               Grid spacing in the normal direction.
 * @param next_curve           Pointer to the next curve in the circular boundary sequence.
 * @param M_p_S_general_param  Cached parameters for the S-patch map (filled by curve_seq_construct_patches).
 * @param M_p_C_param          Cached parameters for the C-patch map.
 * @param J_S_general_param    Cached parameters for the S-patch Jacobian.
 * @param J_C_param            Cached parameters for the C-patch Jacobian.
 */
typedef struct curve {
    scalar_func_t l_1;
    scalar_func_t l_2;
    scalar_func_t l_1_prime;
    scalar_func_t l_2_prime;
    scalar_func_t l_1_dprime;
    scalar_func_t l_2_dprime;

    MKL_INT n;
    MKL_INT n_C_0;
    MKL_INT n_C_1;
    MKL_INT n_S_0;
    MKL_INT n_S_1;

    double h_tan;
    double h_norm;

    curve_t *next_curve;

    /* cached parameters for patch construction */
    M_p_S_general_param_t M_p_S_general_param;
    M_p_C_param_t M_p_C_param;
    J_S_general_param_t J_S_general_param;
    J_C_param_t J_C_param;
} curve_t;

/**
 * @brief Circular singly-linked list of boundary curves defining the domain.
 *
 * @param first_curve  Head of the linked list.
 * @param last_curve   Tail of the linked list (last_curve->next_curve == first_curve).
 * @param n_curves     Number of curves in the sequence.
 */
typedef struct curve_seq {
    curve_t *first_curve;
    curve_t *last_curve;
    MKL_INT n_curves;
} curve_seq_t;

/**
 * @brief Approximates the arc-length of a curve using the trapezoidal rule with 1000 intervals.
 *
 * @param curve  The boundary curve.
 * @return       Approximate arc-length ∫₀¹ |l'(θ)| dθ.
 */
double curve_length(curve_t *curve);

/**
 * @brief Initializes an empty curve sequence.
 *
 * @param curve_seq  Pointer to the sequence to initialize.
 */
void curve_seq_init(curve_seq_t *curve_seq);

/**
 * @brief Appends a curve to the sequence and initializes its discretization.
 *
 * The added curve is inserted at the tail of the circular list.  Grid-point
 * counts n_C_0, n_C_1, n_S_0, n_S_1 are computed from the provided fractions
 * (a value of 0 for a fraction selects the default: 1/10 for C regions and 2/3
 * of the C count for S regions).
 *
 * @param curve_seq    The sequence to append to.
 * @param curve        Pre-allocated curve_t to initialize and insert.
 * @param l_1          x-component of the boundary parametrization.
 * @param l_2          y-component of the boundary parametrization.
 * @param l_1_prime    First derivative of l_1.
 * @param l_2_prime    First derivative of l_2.
 * @param l_1_dprime   Second derivative of l_1.
 * @param l_2_dprime   Second derivative of l_2.
 * @param n            Total grid points along the curve (0 = auto from h_norm).
 * @param frac_n_C_0   Fraction of n in corner overlap at θ = 0 (0 = default).
 * @param frac_n_C_1   Fraction of n in corner overlap at θ = 1 (0 = default).
 * @param frac_n_S_0   Fraction of n_C_0 in smooth overlap at θ = 0 (0 = default).
 * @param frac_n_S_1   Fraction of n_C_1 in smooth overlap at θ = 1 (0 = default).
 * @param h_norm       Normal-direction grid spacing.
 */
void curve_seq_add_curve(curve_seq_t *curve_seq, curve_t *curve, scalar_func_t l_1, scalar_func_t l_2, scalar_func_t l_1_prime, scalar_func_t l_2_prime, scalar_func_t l_1_dprime, scalar_func_t l_2_dprime, MKL_INT n, double frac_n_C_0, double frac_n_C_1, double frac_n_S_0, double frac_n_S_1, double h_norm);

/**
 * @brief Builds all S- and C-patches for every curve in the sequence.
 *
 * For each curve: constructs its S-patch, constructs its C-patch at the
 * trailing corner, then applies partition-of-unity window normalization
 * between each C-patch and the two neighboring S-patches.
 *
 * @param curve_seq      The boundary curve sequence.
 * @param s_patches      Pre-allocated array of n_curves s_patch_t objects.
 * @param c_patches      Pre-allocated array of n_curves c_patch_t objects.
 * @param f_mats         Pre-allocated array of curve_seq_num_f_mats() rd_mat_t wrappers.
 * @param f_mat_points   Pre-allocated data buffer of curve_seq_num_f_mat_points() doubles.
 * @param f              The function f(x, y) to sample.
 * @param d              1D-BTZ matching mesh size.
 * @param eps_xi_eta     Newton convergence tolerance in (ξ, η).
 * @param eps_xy         Newton convergence tolerance in (x, y).
 */
void curve_seq_construct_patches(curve_seq_t *curve_seq, s_patch_t *s_patches, c_patch_t *c_patches, rd_mat_t *f_mats, double *f_mat_points, scalar_func_2D_t f, MKL_INT d, double eps_xi_eta, double eps_xy);

/**
 * @brief Returns the number of rd_mat_t wrappers needed by curve_seq_construct_patches.
 *
 * Each curve contributes 3 matrices: one S-patch, and two (W and L) for its C-patch.
 *
 * @param curve_seq  The boundary curve sequence.
 * @return           3 * n_curves.
 */
MKL_INT curve_seq_num_f_mats(curve_seq_t *curve_seq);

/**
 * @brief Returns the total number of doubles needed for all patch data.
 *
 * @param curve_seq  The boundary curve sequence.
 * @param d          1D-BTZ matching mesh size.
 * @return           Sum of element counts over all S- and C-patch data matrices.
 */
MKL_INT curve_seq_num_f_mat_points(curve_seq_t *curve_seq, MKL_INT d);

/**
 * @brief Returns the number of q_patch_t wrappers needed for all FC patches.
 *
 * Each curve produces 4 FC patches: 1 for the S-patch and 3 (L, W, corner)
 * for the C-patch.
 *
 * @param curve_seq  The boundary curve sequence.
 * @return           4 * n_curves.
 */
MKL_INT curve_seq_num_FC_mats(curve_seq_t *curve_seq);

/**
 * @brief Returns the total number of doubles needed for all FC patch data.
 *
 * @param curve_seq  The boundary curve sequence.
 * @param s_patches  Array of constructed S-patches.
 * @param c_patches  Array of constructed C-patches.
 * @param C          Number of continuation points.
 * @param n_r        Refinement factor.
 * @param d          1D-BTZ matching mesh size.
 * @return           Sum of FC element counts over all patches.
 */
MKL_INT curve_seq_num_FC_points(curve_seq_t *curve_seq, s_patch_t *s_patches, c_patch_t *c_patches, MKL_INT C, MKL_INT n_r, MKL_INT d);

/**
 * @brief Returns the number of points in a refined boundary mesh.
 *
 * Each segment of n-1 intervals is subdivided into n_r sub-intervals, giving
 * (n-1)*n_r + 1 points per curve.
 *
 * @param curve_seq  The boundary curve sequence.
 * @param n_r       Refinement factor of returned boundary mesh.
 * @return           Total number of boundary mesh points.
 */
MKL_INT curve_seq_boundary_mesh_num_el(curve_seq_t *curve_seq, MKL_INT n_r);

/**
 * @brief Samples the boundary curves on a refined mesh and writes (x, y) coordinates.
 *
 * @param curve_seq   The boundary curve sequence.
 * @param n_r         Refinement factor of returned boundary mesh.
 * @param boundary_X  Output column vector of x-coordinates (length from curve_seq_boundary_mesh_num_el).
 * @param boundary_Y  Output column vector of y-coordinates (same length).
 */
void curve_seq_construct_boundary_mesh(curve_seq_t *curve_seq, MKL_INT n_r, rd_mat_t *boundary_X, rd_mat_t *boundary_Y);

#endif
