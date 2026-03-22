#ifndef __CURVE_SEQ_LIB_H__
#define __CURVE_SEQ_LIB_H__

#include <stdlib.h>
#include <mkl.h>

#include "num_linalg_lib.h"
#include "s_patch_lib.h"
#include "c_patch_lib.h"

typedef struct curve curve_t;

typedef struct M_p_S_general_param {
    double xi_diff;
    double xi_0;

    scalar_func_t l_1;
    scalar_func_t l_2;
    scalar_func_t l_1_prime;
    scalar_func_t l_2_prime;
} M_p_S_general_param_t;

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

    // for patch construction
    M_p_S_general_param_t M_p_S_general_param;
    M_p_C_param_t M_p_C_param;
    J_S_general_param_t J_S_general_param;
    J_C_param_t J_C_param;
} curve_t;

typedef struct curve_seq {
    curve_t *first_curve;
    curve_t *last_curve;
    MKL_INT n_curves;
} curve_seq_t;

double curve_length(curve_t *curve);

void curve_seq_init(curve_seq_t *curve_seq);

void curve_seq_add_curve(curve_seq_t *curve_seq, curve_t *curve, scalar_func_t l_1, scalar_func_t l_2, scalar_func_t l_1_prime, scalar_func_t l_2_prime, scalar_func_t l_1_dprime, scalar_func_t l_2_dprime, MKL_INT n, double frac_n_C_0, double frac_n_C_1, double frac_n_S_0, double frac_n_S_1, double h_norm);

void curve_seq_construct_patches(curve_seq_t *curve_seq, s_patch_t *s_patches, c_patch_t *c_patches, rd_mat_t *f_mats, double *f_mat_points, scalar_func_2D_t f, MKL_INT d, double eps_xi_eta, double eps_xy);

MKL_INT curve_seq_num_f_mats(curve_seq_t *curve_seq);

MKL_INT curve_seq_num_f_mat_points(curve_seq_t *curve_seq, MKL_INT d);

MKL_INT curve_seq_num_FC_mats(curve_seq_t *curve_seq);

MKL_INT curve_seq_num_FC_points(curve_seq_t *curve_seq, s_patch_t *s_patches, c_patch_t *c_patches, MKL_INT C, MKL_INT n_r, MKL_INT d);

MKL_INT curve_seq_boundary_mesh_num_el(curve_seq_t *curve_seq, MKL_INT n_r);

void curve_seq_construct_boundary_mesh(curve_seq_t *curve_seq, MKL_INT n_r, rd_mat_t *boundary_X, rd_mat_t *boundary_Y);

#endif