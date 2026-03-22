#include <stdlib.h>
#include <mkl.h>
#include <math.h>
#include <stdio.h>

#include "curve_seq_lib.h"
#include "num_linalg_lib.h"
#include "s_patch_lib.h"
#include "c_patch_lib.h"

void M_p_S_general_handle(rd_mat_t xi, rd_mat_t eta, double H, rd_mat_t *x, rd_mat_t *y, void* extra_param) {
    rd_mat_shape(x, xi.rows, xi.columns);
    rd_mat_shape(y, xi.rows, xi.columns);
    
    // using p for conciseness
    M_p_S_general_param_t *p = (M_p_S_general_param_t*) extra_param;

    for (MKL_INT i = 0; i < xi.rows*xi.columns; i++) {
        double xi_val = xi.mat_data[i];
        double eta_val = eta.mat_data[i];

        double xi_tilde = p->xi_diff*xi_val + p->xi_0;
        double nu_norm = sqrt(pow(p->l_1_prime(xi_tilde), 2) + pow(p->l_2_prime(xi_tilde), 2));

        x->mat_data[i] = p->l_1(xi_tilde) - eta_val*H*p->l_2_prime(xi_tilde)/nu_norm;
        y->mat_data[i] = p->l_2(xi_tilde) + eta_val*H*p->l_1_prime(xi_tilde)/nu_norm;
    }
}

void J_S_general_handle(rd_mat_t v, double H, rd_mat_t *J_vals, void* extra_param) {
    rd_mat_shape(J_vals, 2, 2);

    J_S_general_param_t *p = (J_S_general_param_t*) extra_param;

    double xi = v.mat_data[0];
    double eta = v.mat_data[1];

    double xi_tilde = p->xi_diff*xi + p->xi_0;
    double nu_norm = sqrt(pow(p->l_1_prime(xi_tilde), 2) + pow(p->l_2_prime(xi_tilde), 2));

    J_vals->mat_data[0] = p->xi_diff * (p->l_1_prime(xi_tilde)-eta*H*(p->l_2_dprime(xi_tilde)*pow(nu_norm, 2)-p->l_2_prime(xi_tilde)*(p->l_2_dprime(xi_tilde)*p->l_2_prime(xi_tilde)+p->l_1_dprime(xi_tilde)*p->l_1_prime(xi_tilde)))/pow(nu_norm, 3));
    J_vals->mat_data[1] = p->xi_diff * (p->l_2_prime(xi_tilde)+eta*H*(p->l_1_dprime(xi_tilde)*pow(nu_norm, 2)-p->l_1_prime(xi_tilde)*(p->l_2_dprime(xi_tilde)*p->l_2_prime(xi_tilde)+p->l_1_dprime(xi_tilde)*p->l_1_prime(xi_tilde)))/pow(nu_norm, 3));
    J_vals->mat_data[2] = -H*p->l_2_prime(xi_tilde) / nu_norm;
    J_vals->mat_data[3] = H*p->l_1_prime(xi_tilde) / nu_norm;
}

void M_p_C_handle(rd_mat_t xi, rd_mat_t eta, rd_mat_t *x, rd_mat_t *y, void* extra_param) {
    rd_mat_shape(x, xi.rows, xi.columns);
    rd_mat_shape(y, xi.rows, xi.columns);
    
    // using p for conciseness
    M_p_C_param_t *p = (M_p_C_param_t*) extra_param;

    for (MKL_INT i = 0; i < xi.rows*xi.columns; i++) {
        double xi_val = xi.mat_data[i];
        double eta_val = eta.mat_data[i];

        double xi_tilde = p->xi_diff*xi_val + p->xi_0;
        double eta_tilde = p->eta_diff*eta_val + p->eta_0;

        x->mat_data[i] = p->l_1(xi_tilde) + p->next_curve_l_1(eta_tilde) - p->l_1(1.0);
        y->mat_data[i] = p->l_2(xi_tilde) + p->next_curve_l_2(eta_tilde) - p->l_2(1.0);
    }
}

void J_C_handle(rd_mat_t v, rd_mat_t *J_vals, void* extra_param) {
    rd_mat_shape(J_vals, 2, 2);

    J_C_param_t *p = (J_C_param_t*) extra_param;

    double xi = v.mat_data[0];
    double eta = v.mat_data[1];

    double xi_tilde = p->xi_diff*xi + p->xi_0;
    double eta_tilde = p->eta_diff*eta + p->eta_0;

    J_vals->mat_data[0] = p->xi_diff * (p->l_1_prime(xi_tilde));
    J_vals->mat_data[1] = p->xi_diff * (p->l_2_prime(xi_tilde));
    J_vals->mat_data[2] = p->eta_diff *(p->next_curve_l_1_prime(eta_tilde));
    J_vals->mat_data[3] = p->eta_diff *(p->next_curve_l_2_prime(eta_tilde));
}

double curve_length(curve_t *curve) {
    MKL_INT n = 1000;
    double h = 1.0/((double) n);
    double length = 0.5*(sqrt(pow(curve->l_1_prime(0), 2) + pow(curve->l_2_prime(0), 2)) + sqrt(pow(curve->l_1_prime(1), 2) + pow(curve->l_2_prime(1), 2)));

    for (int i = 1; i < n; i++) {
        double t = i*h;
        length += sqrt(pow(curve->l_1_prime(t), 2) + pow(curve->l_2_prime(t), 2));
    }

    return h*length;
}

void curve_init(curve_t *curve, scalar_func_t l_1, scalar_func_t l_2, scalar_func_t l_1_prime, scalar_func_t l_2_prime, scalar_func_t l_1_dprime, scalar_func_t l_2_dprime, MKL_INT n, double frac_n_C_0, double frac_n_C_1, double frac_n_S_0, double frac_n_S_1, double h_norm, curve_t *next_curve) {
    curve->l_1 = l_1;
    curve->l_2 = l_2;
    curve->l_1_prime = l_1_prime;
    curve->l_2_prime = l_2_prime;
    curve->l_1_dprime = l_1_dprime;
    curve->l_2_dprime = l_2_dprime;

    if(n == 0) {
        curve->n = ceil(curve_length(curve) / h_norm) + 1;
    } else {
        curve->n = n;
    }
    
    if(frac_n_C_0 == 0) {
        curve->n_C_0 = ceil(1.0/10.0*curve->n);
    } else {
        curve->n_C_0 = ceil(frac_n_C_0*curve->n);
    }

    if(frac_n_C_1 == 0) {
        curve->n_C_1 = ceil(1.0/10.0*curve->n);
    } else {
        curve->n_C_1 = ceil(frac_n_C_1*curve->n);
    }

    if(frac_n_S_0 == 0) {
        curve->n_S_0 = ceil(2.0/3.0*curve->n_C_0);
    } else {
        curve->n_S_0 = ceil(frac_n_S_0*curve->n_C_0);
    }

    if(frac_n_S_1 == 0) {
        curve->n_S_1 = ceil(2.0/3.0*curve->n_C_1);
    } else {
        curve->n_S_1 = ceil(frac_n_S_1*curve->n_C_1);
    }

    curve->h_tan = 1.0/(curve->n-1);
    curve->h_norm = h_norm;

    if(next_curve == NULL) {
        curve->next_curve = curve;
    } else {
        curve->next_curve = next_curve;
    }
}

MKL_INT curve_S_patch_num_el(curve_t *curve, double d) {
    MKL_INT s_patch_n_xi = curve->n - (curve->n_C_1-curve->n_S_1) - (curve->n_C_0-curve->n_S_0);
    return s_patch_n_xi * d;
}

MKL_INT curve_construct_S_patch(curve_t *curve, s_patch_t *s_patch, rd_mat_t *s_patch_f_XY, double *data_stack, scalar_func_2D_t f, MKL_INT d, double eps_xi_eta, double eps_xy) {
    s_patch_f_XY->mat_data = data_stack;

    double xi_diff = 1-(curve->n_C_1-curve->n_S_1)*curve->h_tan - (curve->n_C_0-curve->n_S_0)*curve->h_tan;
    double xi_0 = (curve->n_C_0 - curve->n_S_0) * curve->h_tan;

    curve->M_p_S_general_param = (M_p_S_general_param_t) {xi_diff, xi_0, curve->l_1, curve->l_2, curve->l_1_prime, curve->l_2_prime};
    M_p_general_t M_p_general = {(M_p_general_handle_t) M_p_S_general_handle, (void*) &(curve->M_p_S_general_param)};

    curve->J_S_general_param = (J_S_general_param_t) {xi_diff, xi_0, curve->l_1, curve->l_2, curve->l_1_prime, curve->l_2_prime, curve->l_1_dprime, curve->l_2_dprime};
    J_general_t J_general = {(J_general_handle_t) J_S_general_handle, (void*) &(curve->J_S_general_param)};

    MKL_INT s_patch_n_xi = curve->n - (curve->n_C_1-curve->n_S_1) - (curve->n_C_0-curve->n_S_0);
    s_patch_init(s_patch, M_p_general, J_general, curve->h_norm, eps_xi_eta, eps_xy, s_patch_n_xi, d, s_patch_f_XY);

    q_patch_evaluate_f(&(s_patch->Q), (scalar_func_2D_t) f);

    return curve_S_patch_num_el(curve, d);
}

MKL_INT curve_C2_patch_f_L_num_el(curve_t *curve, MKL_INT d) {
    return (curve->next_curve->n_C_0 - (d-1)) * d ;
}

MKL_INT curve_C2_patch_f_W_num_el(curve_t *curve, MKL_INT d) {
    return (curve->n_C_1) * d;
}

MKL_INT curve_C1_patch_f_L_num_el(curve_t *curve, MKL_INT d) {
    return (curve->next_curve->n_C_0 + (d-1)) * d ;
}

MKL_INT curve_C1_patch_f_W_num_el(curve_t *curve, MKL_INT d) {
    return (curve->n_C_1) * d;
}

MKL_INT curve_construct_C_patch(curve_t *curve, c_patch_t *c_patch, rd_mat_t *c_patch_f_W, rd_mat_t *c_patch_f_L, double *data_stack, scalar_func_2D_t f, MKL_INT d, double eps_xi_eta, double eps_xy) {
    double curr_v_xi = curve->l_1(1.0) - curve->l_1(1.0-1.0/(curve->n-1));
    double curr_v_eta = curve->l_2(1.0) - curve->l_2(1.0-1.0/(curve->n-1));
    double next_v_xi = curve->next_curve->l_1(1.0/(curve->next_curve->n-1)) - curve->l_1(1.0);
    double next_v_eta = curve->next_curve->l_2(1.0/(curve->next_curve->n-1)) - curve->l_2(1.0);
    
    MKL_INT total_el;
    c_patch_f_W->mat_data = data_stack;
    if(curr_v_xi*next_v_eta - curr_v_eta*next_v_xi >= 0) {
        c_patch_f_L->mat_data = data_stack + curve_C2_patch_f_W_num_el(curve, d);

        double xi_diff = -(curve->n_C_1-1)*curve->h_tan;
        double eta_diff = (curve->next_curve->n_C_0-1)*curve->next_curve->h_tan;
        double xi_0 = 1;
        double eta_0 = 0;

        curve->M_p_C_param = (M_p_C_param_t) {xi_diff, xi_0, eta_diff, eta_0, curve->l_1, curve->l_2, curve->next_curve->l_1, curve->next_curve->l_2};
        M_p_t M_p = (M_p_t) {(M_p_handle_t) M_p_C_handle, (void*) &(curve->M_p_C_param)};

        curve->J_C_param = (J_C_param_t) {xi_diff, xi_0, eta_diff, eta_0, curve->l_1_prime, curve->l_2_prime, curve->next_curve->l_1_prime, curve->next_curve->l_2_prime};
        J_t J = (J_t) {(J_handle_t) J_C_handle, (void*) &(curve->J_C_param)};

        c2_patch_init(c_patch, M_p, J, eps_xi_eta, eps_xy, curve->n_C_1, curve->next_curve->n_C_0, d, c_patch_f_L, c_patch_f_W);

        total_el = curve_C2_patch_f_L_num_el(curve, d) + curve_C2_patch_f_W_num_el(curve, d);
    } else {
        c_patch_f_L->mat_data = data_stack + curve_C1_patch_f_W_num_el(curve, d);

        double xi_diff = 2*(curve->n_C_1-1)*curve->h_tan;
        double eta_diff = -2*(curve->next_curve->n_C_0-1)*curve->next_curve->h_tan;
        double xi_0 = 1-(curve->n_C_1-1)*curve->h_tan;
        double eta_0 = (curve->next_curve->n_C_0-1)*curve->next_curve->h_tan;

        curve->M_p_C_param = (M_p_C_param_t) {xi_diff, xi_0, eta_diff, eta_0, curve->l_1, curve->l_2, curve->next_curve->l_1, curve->next_curve->l_2};
        M_p_t M_p = (M_p_t) {(M_p_handle_t) M_p_C_handle, (void*) &(curve->M_p_C_param)};

        curve->J_C_param = (J_C_param_t) {xi_diff, xi_0, eta_diff, eta_0, curve->l_1_prime, curve->l_2_prime, curve->next_curve->l_1_prime, curve->next_curve->l_2_prime};
        J_t J = (J_t) {(J_handle_t) J_C_handle, (void*) &(curve->J_C_param)};

        c1_patch_init(c_patch, M_p, J, eps_xi_eta, eps_xy, curve->n_C_1*2-1, curve->next_curve->n_C_0*2-1, d, c_patch_f_L, c_patch_f_W);
        total_el = curve_C1_patch_f_L_num_el(curve, d) + curve_C1_patch_f_W_num_el(curve, d);
    }

    q_patch_evaluate_f(&(c_patch->L), f);
    q_patch_evaluate_f(&(c_patch->W), f);

    return total_el;
}

void curve_seq_init(curve_seq_t *curve_seq) {
    curve_seq->first_curve = NULL;
    curve_seq->last_curve = NULL;
    curve_seq->n_curves = 0;
}

void curve_seq_add_curve(curve_seq_t *curve_seq, curve_t *curve, scalar_func_t l_1, scalar_func_t l_2, scalar_func_t l_1_prime, scalar_func_t l_2_prime, scalar_func_t l_1_dprime, scalar_func_t l_2_dprime, MKL_INT n, double frac_n_C_0, double frac_n_C_1, double frac_n_S_0, double frac_n_S_1, double h_norm) {
    curve_init(curve, l_1, l_2, l_1_prime, l_2_prime, l_1_dprime, l_2_dprime, n, frac_n_C_0, frac_n_C_1, frac_n_S_0, frac_n_S_1, h_norm, curve_seq->first_curve);
    if(curve_seq->first_curve == NULL) {
        curve_seq->first_curve = curve;
        curve_seq->last_curve = curve;
    }

    curve_seq->last_curve->next_curve = curve;
    curve_seq->last_curve = curve;            
    curve_seq->n_curves += 1;
}

MKL_INT curve_seq_num_f_mats(curve_seq_t *curve_seq) {
    return curve_seq->n_curves*3;
}

MKL_INT curve_seq_num_f_mat_points(curve_seq_t *curve_seq, MKL_INT d) {
    curve_t *curr = curve_seq->first_curve;
    MKL_INT total_el = 0;
    for (int i = 0; i < curve_seq->n_curves; i++) {
        total_el += curve_S_patch_num_el(curr, d);

        double curr_v_xi = curr->l_1(1.0) - curr->l_1(1.0-1.0/(curr->n-1));
        double curr_v_eta = curr->l_2(1.0) - curr->l_2(1.0-1.0/(curr->n-1));
        double next_v_xi = curr->next_curve->l_1(1.0/(curr->next_curve->n-1)) - curr->l_1(1.0);
        double next_v_eta = curr->next_curve->l_2(1.0/(curr->next_curve->n-1)) - curr->l_2(1.0);

        if(curr_v_xi*next_v_eta - curr_v_eta*next_v_xi >= 0) {
            total_el += curve_C2_patch_f_L_num_el(curr, d) + curve_C2_patch_f_W_num_el(curr, d);
        } else {
            total_el += curve_C1_patch_f_L_num_el(curr, d) + curve_C1_patch_f_W_num_el(curr, d);
        }
        curr = curr->next_curve;
    }

    return total_el;
}

void curve_seq_construct_patches(curve_seq_t *curve_seq, s_patch_t *s_patches, c_patch_t *c_patches, rd_mat_t *f_mats, double *f_mat_points, scalar_func_2D_t f, MKL_INT d, double eps_xi_eta, double eps_xy) {
    curve_t *curr_curve = curve_seq->first_curve;
    c_patch_t *curr_c_patch = c_patches;
    s_patch_t *curr_s_patch = s_patches;
    rd_mat_t *curr_f_mat = f_mats;
    double *curr_f_mat_point = f_mat_points;

    s_patch_t *prev_s_patch;
    c_patch_t *prev_c_patch;
    for (int i = 0; i < curve_seq->n_curves; i++) {
        curr_f_mat_point += curve_construct_S_patch(curr_curve, curr_s_patch, curr_f_mat, curr_f_mat_point, f, d, eps_xi_eta, eps_xy);
        curr_f_mat += 1;

        curr_f_mat_point += curve_construct_C_patch(curr_curve, curr_c_patch, curr_f_mat, curr_f_mat+1, curr_f_mat_point, f, d, eps_xi_eta, eps_xy);
        curr_f_mat += 2;

        if (i != 0) {
            if (prev_c_patch->c_patch_type == C1) {
                c1_patch_apply_w_W(prev_c_patch, prev_s_patch);
                c1_patch_apply_w_L(prev_c_patch, curr_s_patch);
            } else {
                c2_patch_apply_w_W(prev_c_patch, prev_s_patch);
                c2_patch_apply_w_L(prev_c_patch, curr_s_patch);
            }
        }
        
        prev_s_patch = curr_s_patch;
        prev_c_patch = curr_c_patch;
        curr_curve = curr_curve->next_curve;
        curr_c_patch += 1;
        curr_s_patch += 1;
    }
    if (prev_c_patch->c_patch_type == C1) {
        c1_patch_apply_w_W(prev_c_patch, prev_s_patch);
        c1_patch_apply_w_L(prev_c_patch, s_patches);
    } else {
        c2_patch_apply_w_W(prev_c_patch, prev_s_patch);
        c2_patch_apply_w_L(prev_c_patch, s_patches);
    }
}

MKL_INT curve_seq_num_FC_mats(curve_seq_t *curve_seq) {
    return 4*curve_seq->n_curves;
}

MKL_INT curve_seq_num_FC_points(curve_seq_t *curve_seq, s_patch_t *s_patches, c_patch_t *c_patches, MKL_INT C, MKL_INT n_r, MKL_INT d) {
    MKL_INT total_points = 0;
    for (int i = 0; i < curve_seq->n_curves; i++) {
        total_points += s_patch_FC_num_el(s_patches+i, C, n_r);
        total_points += c_patch_FC_corner_num_el(C, n_r);
        
        if (c_patches[i].c_patch_type == C2) {
            total_points += c2_patch_FC_W_num_el(c_patches+i, C, n_r) + c2_patch_FC_L_num_el(c_patches+i, C, n_r, d);
        } else {
            total_points += c1_patch_FC_W_num_el(c_patches+i, C, n_r) + c1_patch_FC_L_num_el(c_patches+i, C, n_r, d);
        }
        
    }

    return total_points;
}

MKL_INT curve_seq_boundary_mesh_num_el(curve_seq_t *curve_seq, MKL_INT n_r) {
    curve_t *curr_curve = curve_seq->first_curve;
    MKL_INT n_points = 0;
    for (int i = 0; i < curve_seq->n_curves; i++) {
        n_points += (curr_curve->n-1)*n_r + 1;
        curr_curve = curr_curve->next_curve;
    }

    return n_points;
}

void curve_seq_construct_boundary_mesh(curve_seq_t *curve_seq, MKL_INT n_r, rd_mat_t *boundary_X, rd_mat_t *boundary_Y) {
    MKL_INT n_points = curve_seq_boundary_mesh_num_el(curve_seq, n_r);
    rd_mat_shape(boundary_X, n_points, 1);
    rd_mat_shape(boundary_Y, n_points, 1);

    MKL_INT curr_idx = 0;
    curve_t *curr_curve = curve_seq->first_curve;
    for (int i = 0; i < curve_seq->n_curves; i++) {
        MKL_INT n_curve = (curr_curve->n-1)*n_r+1;

        double theta_mesh_data[n_curve];
        rd_mat_t theta_mesh = rd_mat_init(theta_mesh_data, n_curve, 1);
        rd_linspace(0, 1, n_curve, &theta_mesh);

        for (int j = 0; j < n_curve; j++) {
            boundary_X->mat_data[curr_idx+j] = curr_curve->l_1(theta_mesh_data[j]);
            boundary_Y->mat_data[curr_idx+j] = curr_curve->l_2(theta_mesh_data[j]);
        }

        curr_idx += n_curve;
        curr_curve = curr_curve->next_curve;
    }
}