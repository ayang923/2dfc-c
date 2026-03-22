#include <stdlib.h>
#include<mkl.h>
#include <math.h>
#include <stdio.h>

#include "c_patch_lib.h"
#include "num_linalg_lib.h"
#include "q_patch_lib.h"
#include "fc_lib.h"

void c2_patch_init(c_patch_t *c_patch, M_p_t M_p, J_t J, double eps_xi_eta, double eps_xy, MKL_INT n_xi, MKL_INT n_eta, MKL_INT d, rd_mat_t *f_L, rd_mat_t *f_W) {
    double h_xi = 1.0/(n_xi-1);
    double h_eta = 1.0/(n_eta-1);
    q_patch_init(&(c_patch->W), M_p, J, eps_xi_eta, eps_xy, n_xi, d, 0.0, 1.0, 0.0, (d-1)*h_eta, f_W);
    q_patch_init(&(c_patch->L), M_p, J, eps_xi_eta, eps_xy, d, n_eta-d+1, 0.0, (d-1)*h_xi, (d-1)*h_eta, 1.0, f_L);
    c_patch->c_patch_type = C2;
}

void c1_patch_init(c_patch_t *c_patch, M_p_t M_p, J_t J, double eps_xi_eta, double eps_xy, MKL_INT n_xi, MKL_INT n_eta, MKL_INT d, rd_mat_t *f_L, rd_mat_t *f_W) {
    double h_xi = 1.0/(n_xi-1);
    double h_eta = 1.0/(n_eta-1);
    q_patch_init(&(c_patch->W), M_p, J, eps_xi_eta, eps_xy, (n_xi+1)/2, d, 0.0, 1.0/2.0, 1.0/2.0, 1.0/2.0+(d-1)*h_eta, f_W);
    q_patch_init(&(c_patch->L), M_p, J, eps_xi_eta, eps_xy, d, (n_eta+1)/2+(d-1), 1.0/2.0, 1.0/2.0+(d-1)*h_xi, 0.0, 1.0/2.0+(d-1)*h_eta, f_L);
    c_patch->c_patch_type = C1;
}

void c1_patch_apply_w_W(c_patch_t *c_patch, s_patch_t *window_patch_W) {
    q_patch_apply_w_normalization_xi_left(&(c_patch->W), &(window_patch_W->Q));
}

void c1_patch_apply_w_L(c_patch_t *c_patch, s_patch_t *window_patch_L) {
    q_patch_apply_w_normalization_eta_down(&(c_patch->L), &(window_patch_L->Q));
}

void c2_patch_apply_w_W(c_patch_t *c_patch, s_patch_t *window_patch_W) {
    q_patch_apply_w_normalization_xi_right(&(c_patch->W), &(window_patch_W->Q));
}

void c2_patch_apply_w_L(c_patch_t *c_patch, s_patch_t *window_patch_L) {
    q_patch_apply_w_normalization_eta_up(&(c_patch->L), &(window_patch_L->Q));
}

MKL_INT c1_patch_FC_W_num_el(c_patch_t *c_patch, MKL_INT C, MKL_INT n_r) {
    return (c_patch->W.n_xi-C) * (C*n_r+1);
}

MKL_INT c2_patch_FC_W_num_el(c_patch_t *c_patch, MKL_INT C, MKL_INT n_r) {
    return (C*n_r+1) * (c_patch->W.n_xi);
}

MKL_INT c1_patch_FC_L_num_el(c_patch_t *c_patch, MKL_INT C, MKL_INT n_r, MKL_INT d) {
    return (c_patch->L.n_eta - (d-1)) * (C*n_r+1);
}

MKL_INT c2_patch_FC_L_num_el(c_patch_t *c_patch, MKL_INT C, MKL_INT n_r, MKL_INT d) {
    return (C*n_r+1) * (c_patch->L.n_eta+d-1);
}

MKL_INT c_patch_FC_corner_num_el(MKL_INT C, MKL_INT n_r) {
    return pow(C*n_r + 1, 2);
}

MKL_INT c2_patch_FC(c_patch_t *c_patch, MKL_INT C, MKL_INT n_r, MKL_INT d, rd_mat_t A, rd_mat_t Q, q_patch_t* c2_fcont_patch_L, q_patch_t *c2_fcont_patch_W, q_patch_t *c2_fcont_patch_corner, rd_mat_t *f_L, rd_mat_t *f_W, rd_mat_t *f_corner, double *data_stack) {
    f_L->mat_data = data_stack;
    f_W->mat_data = f_L->mat_data + c2_patch_FC_L_num_el(c_patch, C, n_r, d);
    f_corner->mat_data = f_W->mat_data + c2_patch_FC_W_num_el(c_patch, C, n_r);
    
    q_patch_init(c2_fcont_patch_L, c_patch->L.M_p, c_patch->L.J, c_patch->L.eps_xi_eta, c_patch->L.eps_xy, C*n_r+1, c_patch->L.n_eta+d-1, -C*c_patch->L.h_xi, 0, 0, 1, f_L);
    q_patch_init(c2_fcont_patch_W, c_patch->W.M_p, c_patch->W.J, c_patch->W.eps_xi_eta, c_patch->W.eps_xy, c_patch->W.n_xi, C*n_r+1, 0, 1, -C*c_patch->L.h_eta, 0, f_W);
    q_patch_init(c2_fcont_patch_corner, c_patch->W.M_p, c_patch->W.J, c_patch->W.eps_xi_eta, c_patch->W.eps_xy, C*n_r+1, C*n_r+1, -C*c_patch->W.h_xi, 0, -C*c_patch->W.h_eta, 0, f_corner);

    fcont_gram_blend_S(*(c_patch->W.f_XY), d, A, Q, c2_fcont_patch_W->f_XY);

    // computing matrix transpose
    double fcont_W_T_data[c2_fcont_patch_W->f_XY->rows*c2_fcont_patch_W->f_XY->columns];
    mkl_domatcopy('c', 't', c2_fcont_patch_W->f_XY->rows, c2_fcont_patch_W->f_XY->columns, 1.0, c2_fcont_patch_W->f_XY->mat_data, c2_fcont_patch_W->f_XY->rows, fcont_W_T_data, c2_fcont_patch_W->f_XY->columns);

    rd_mat_t fcont_W_T = rd_mat_init(fcont_W_T_data, c2_fcont_patch_W->f_XY->columns, c2_fcont_patch_W->f_XY->rows);

    fcont_gram_blend_S(fcont_W_T, d, A, Q, c2_fcont_patch_corner->f_XY);
    mkl_dimatcopy('c', 't', C*n_r+1, C*n_r+1, 1.0, c2_fcont_patch_corner->f_XY->mat_data, C*n_r+1, C*n_r+1);

    // constructing matrix for L fcont
    double f_L_combined_data[c2_patch_FC_L_num_el(c_patch, C, n_r, d)];
    double *curr_f_W = c_patch->W.f_XY->mat_data;
    double *curr_f_L = c_patch->L.f_XY->mat_data+1;
    double *curr_combined = f_L_combined_data;
    for (int i = 0; i < d; i++) {
        cblas_dcopy(d, curr_f_W, 1, curr_combined, 1);
        curr_combined += d;
        curr_f_W += d;
        cblas_dcopy(c_patch->L.n_eta-1, curr_f_L, 1, curr_combined, 1);
        curr_combined += c_patch->L.n_eta-1;
        curr_f_L += c_patch->L.n_eta;
    }

    mkl_dimatcopy('c', 't', c_patch->L.n_eta+d-1, d, 1.0, f_L_combined_data, c_patch->L.n_eta+d-1, d);
    rd_mat_t f_L_combined = rd_mat_init(f_L_combined_data, d, c_patch->L.n_eta+d-1);
    fcont_gram_blend_S(f_L_combined, d, A, Q, c2_fcont_patch_L->f_XY);

    mkl_dimatcopy('c', 't', C*n_r+1, c_patch->L.n_eta+d-1, 1.0, c2_fcont_patch_L->f_XY->mat_data, C*n_r+1, c_patch->L.n_eta+d-1);
    rd_mat_shape(c2_fcont_patch_L->f_XY, c_patch->L.n_eta+d-1, C*n_r+1);

    return c_patch_FC_corner_num_el(C, n_r) + c2_patch_FC_L_num_el(c_patch, C, n_r, d) + c2_patch_FC_W_num_el(c_patch, C, n_r);
}

MKL_INT c1_patch_FC(c_patch_t *c_patch, MKL_INT C, MKL_INT n_r, MKL_INT d, rd_mat_t A, rd_mat_t Q, MKL_INT M, q_patch_t* c1_fcont_patch_L, q_patch_t *c1_fcont_patch_W, q_patch_t *c1_fcont_patch_corner, rd_mat_t *f_L, rd_mat_t *f_W, rd_mat_t *f_corner, double *data_stack) {
    f_L->mat_data = data_stack;
    f_W->mat_data = f_L->mat_data + c1_patch_FC_L_num_el(c_patch, C, n_r, d);
    f_corner->mat_data = f_W->mat_data + c1_patch_FC_W_num_el(c_patch, C, n_r);

    q_patch_init(c1_fcont_patch_L, c_patch->L.M_p, c_patch->L.J, c_patch->L.eps_xi_eta, c_patch->L.eps_xy, C*n_r+1, c_patch->L.n_eta-(d-1), 0.5-C*c_patch->L.h_xi, 0.5, 0, 0.5, f_L);
    q_patch_init(c1_fcont_patch_W, c_patch->W.M_p, c_patch->W.J, c_patch->W.eps_xi_eta, c_patch->W.eps_xy, c_patch->W.n_xi-C, C*n_r+1, 0, 0.5-C*c_patch->W.h_xi, 0.5-C*c_patch->W.h_eta, 0.5, f_W);
    q_patch_init(c1_fcont_patch_corner, c_patch->W.M_p, c_patch->W.J, c_patch->W.eps_xi_eta, c_patch->W.eps_xy, C*n_r+1, C*n_r+1, 0.5-C*c_patch->W.h_xi, 0.5, 0.5-C*c_patch->W.h_eta, 0.5, f_corner);

    double L_f_XY_T_data[c_patch->L.n_xi * c_patch->L.n_eta];
    rd_mat_t L_f_XY_T = rd_mat_init(L_f_XY_T_data, c_patch->L.n_xi, c_patch->L.n_eta);
    mkl_domatcopy('c', 't', c_patch->L.n_eta, c_patch->L.n_xi, 1.0, c_patch->L.f_XY->mat_data, c_patch->L.n_eta, L_f_XY_T_data, c_patch->L.n_xi);

    double L_fcont_data[(C*n_r+1)*c_patch->L.n_eta];
    rd_mat_t L_fcont = rd_mat_init_no_shape(L_fcont_data);
    fcont_gram_blend_S(L_f_XY_T, d, A, Q, &L_fcont);

    double L_fcont_excess[d*(C*n_r+1)];
    
    mkl_dimatcopy('c', 't', C*n_r+1, c_patch->L.n_eta, 1.0, L_fcont_data, C*n_r+1, c_patch->L.n_eta);
    rd_mat_shape(&L_fcont, c_patch->L.n_eta, C*n_r+1);

    double *curr_L_patch_data = f_L->mat_data;
    double *curr_L_fcont_data = L_fcont_data;
    double *curr_L_fcont_excess = L_fcont_excess;
    for (int i = 0; i < L_fcont.columns; i++) {
        cblas_dcopy(c_patch->L.n_eta-(d-1), curr_L_fcont_data, 1, curr_L_patch_data, 1);
        cblas_dcopy(d, curr_L_fcont_data+c_patch->L.n_eta-d, 1, curr_L_fcont_excess, 1);
        curr_L_patch_data += f_L->rows;
        curr_L_fcont_data += L_fcont.rows;
        curr_L_fcont_excess += d;
    }

    if (0.5-C*c_patch->L.h_xi <= 0) {
        printf("Please refine mesh of C1 Patch!!!\n");
    }
    else {
        double W_refined_f_XY[c_patch->W.n_eta*(C*n_r+1)];
        rd_mat_t W_unrefined_f_XY = rd_mat_init(c_patch->W.f_XY->mat_data, c_patch->W.n_eta, c_patch->W.n_xi-C);
        fcont_gram_blend_S(W_unrefined_f_XY, d, A, Q, c1_fcont_patch_W->f_XY);

        double xi_refined_mesh_data[C*n_r+1];
        rd_mat_t xi_refined_mesh = rd_mat_init(xi_refined_mesh_data, C*n_r+1, 1);
        rd_linspace(0.5-C*c_patch->L.h_xi, 0.5, C*n_r+1, &xi_refined_mesh);


        //copying boundary
        cblas_dcopy(c_patch->W.n_eta, c_patch->W.f_XY->mat_data + (c_patch->W.n_eta) * (c_patch->W.n_xi-1), 1, W_refined_f_XY + C*n_r*c_patch->W.n_eta, 1);
        MKL_INT row_num = c_patch->W.n_eta;
        MKL_INT col_num = c_patch->W.n_xi-C-2;
        MKL_INT half_M = M/2;
        MKL_INT interpol_xi_j_mesh_data[M];
        ri_mat_t interpol_xi_j_mesh = ri_mat_init(interpol_xi_j_mesh_data, M, 1);
        double interpol_xi_mesh_data[M];
        rd_mat_t interpol_xi_mesh = rd_mat_init(interpol_xi_mesh_data, M, 1);
        double interpol_W_val_data[M];
        rd_mat_t interpol_W_val = rd_mat_init(interpol_W_val_data, M, 1);

        for (int i = 0; i < c_patch->W.n_eta*(C); i++) {
            if (row_num == c_patch->W.n_eta) {
                row_num = 0;
                col_num += 1;
                if (M % 2 != 0) {
                    ri_range(col_num-half_M, 1, col_num+half_M, &interpol_xi_j_mesh);
                }
                else {
                    ri_range(col_num-half_M+1, 1, col_num+half_M, &interpol_xi_j_mesh);
                }
                shift_idx_mesh(&interpol_xi_j_mesh, 0, c_patch->W.n_xi-1);
                for (int mesh_idx = 0; mesh_idx < M; mesh_idx++) {
                    // no additive constant because W start is 0
                    interpol_xi_mesh_data[mesh_idx] = interpol_xi_j_mesh_data[mesh_idx]*c_patch->W.h_xi;
                }
            }
            for (int mesh_idx = 0; mesh_idx < M; mesh_idx++) {
                MKL_INT idx = sub2ind(c_patch->W.n_eta, c_patch->W.n_xi, (sub_t) {row_num, interpol_xi_j_mesh_data[mesh_idx]});
                interpol_W_val_data[mesh_idx] = c_patch->W.f_XY->mat_data[idx];
            }

            for (int refined_mesh_idx = 0; refined_mesh_idx < n_r; refined_mesh_idx++) {
                MKL_INT W_refined_f_XY_idx = sub2ind(c_patch->W.n_eta, C*n_r+1, (sub_t) {row_num, (col_num-(c_patch->W.n_xi-C-1))*n_r+refined_mesh_idx});
                double xi = xi_refined_mesh_data[(col_num-(c_patch->W.n_xi-C-1))*n_r+refined_mesh_idx];

                W_refined_f_XY[W_refined_f_XY_idx] = barylag(interpol_xi_mesh, interpol_W_val, xi);
            }
            row_num += 1;
        }

        double W_minus_fcont_data[c_patch->W.n_eta*(C*n_r+1)];
        rd_mat_t W_minus_fcont = rd_mat_init(W_minus_fcont_data, c_patch->W.n_eta, C*n_r+1);

        vdSub(d*(C*n_r+1), W_refined_f_XY, L_fcont_excess, W_minus_fcont_data);

        fcont_gram_blend_S(W_minus_fcont, d, A, Q, c1_fcont_patch_corner->f_XY);
    }

    return c1_patch_FC_L_num_el(c_patch, C, n_r, d) + c1_patch_FC_W_num_el(c_patch, C, n_r) + c_patch_FC_corner_num_el(C, n_r);
}
