
#include "s_patch_lib.h"
#include  "q_patch_lib.h"
#include "fc_lib.h"

#include <stdio.h>

void s_M_p_wrapper_function(rd_mat_t xi, rd_mat_t eta, rd_mat_t *x, rd_mat_t *y, void* extra_param) {
    s_patch_t *s_patch = (s_patch_t*) extra_param;
    s_patch->M_p_general.M_p_general_handle(xi, eta, s_patch->H, x, y, s_patch->M_p_general.extra_param);
}

void s_J_wrapper_function(rd_mat_t v, rd_mat_t *J_vals, void* extra_param) {
    s_patch_t *s_patch = (s_patch_t*) extra_param;
    s_patch->J_general.J_general_handle(v, s_patch->H, J_vals, s_patch->J_general.extra_param);
}

void s_patch_init(s_patch_t *s_patch, M_p_general_t M_p_general, J_general_t J_general, double h, double eps_xi_eta, double eps_xy, MKL_INT n_xi, MKL_INT d, rd_mat_t *f_XY) {    
    s_patch->h = h;
    s_patch->H = 1;
    s_patch->M_p_general = M_p_general;
    s_patch->J_general = J_general;

    q_patch_init(&(s_patch->Q), (M_p_t) {(M_p_handle_t) s_M_p_wrapper_function, (void*) s_patch}, (J_t) {(J_handle_t) s_J_wrapper_function, (void*) s_patch}, eps_xi_eta, eps_xy, n_xi, d, 0.0, 1.0, 0.0, (d-1)*h, f_XY);
}

MKL_INT s_patch_FC_num_el(s_patch_t *s_patch, MKL_INT C, MKL_INT n_r) {
    return (n_r*C+1)*(s_patch->Q.n_xi);
}

MKL_INT s_patch_FC(s_patch_t *s_patch, MKL_INT C, MKL_INT n_r, MKL_INT d, rd_mat_t A, rd_mat_t Q, q_patch_t* s_patch_FC, rd_mat_t *f_FC, double *data_stack) {
    f_FC->mat_data = data_stack;
    q_patch_t *sq_patch = &(s_patch->Q);
    q_patch_init(s_patch_FC, sq_patch->M_p, sq_patch->J, sq_patch->eps_xi_eta, sq_patch->eps_xy, sq_patch->n_xi, C*n_r+1, sq_patch->xi_start, sq_patch->xi_end, sq_patch->eta_start-C*sq_patch->h_eta, sq_patch->eta_start, f_FC);

    fcont_gram_blend_S(*(s_patch->Q.f_XY), d, A, Q, s_patch_FC->f_XY);

    return s_patch_FC_num_el(s_patch, C, n_r);
}

