#include <stdio.h>
#include <mkl.h>
#include <math.h>

#include "q_patch_lib.h"
#include "num_linalg_lib.h"
#include "fc_lib.h"

typedef struct M_p_C2_extra_param {
    double theta_A;
    double theta_B;
    double theta_C;
} M_p_C2_extra_param_t;

typedef struct J_C2_extra_param {
    double theta_A;
    double theta_B;
} J_C2_extra_param_t;

void M_p_C2(rd_mat_t xi, rd_mat_t eta, rd_mat_t *x, rd_mat_t *y, void* extra_param) {
    M_p_C2_extra_param_t* C2_extra_param = (M_p_C2_extra_param_t*) extra_param;

    // assumes xi and eta are of same size, using for loopos for ease of implementation
    for (MKL_INT i = 0; i < xi.rows*xi.columns; i++) {
        double l_A = xi.mat_data[i]*(C2_extra_param->theta_A - C2_extra_param->theta_C) + C2_extra_param->theta_C;
        double l_B = eta.mat_data[i]*(C2_extra_param->theta_B - C2_extra_param->theta_C - 2*M_PI) + C2_extra_param->theta_C + 2*M_PI;

        // printf("(%f, %f)\n", l_A, l_B);

        x->mat_data[i] = 2*sin(l_A/2) + 2*sin(l_B/2) - 2*sin(C2_extra_param->theta_C/2);
        y->mat_data[i] = -sin(l_A) - sin(l_B) - sin(C2_extra_param->theta_C);
    }
}

void J_C2(rd_mat_t v, rd_mat_t *J_vals, void* extra_param) {
    J_C2_extra_param_t *C2_extra_param = (J_C2_extra_param_t*) extra_param;
    double theta_A = C2_extra_param->theta_A;
    double theta_B = C2_extra_param->theta_B;

    // assumes v is a 2x1 vector
    J_vals->rows = 2;
    J_vals->columns = 2;

    J_vals->mat_data[0] = theta_A*cos(v.mat_data[0]*theta_A/2);
    J_vals->mat_data[1] = -theta_A*cos(v.mat_data[0]*theta_A);
    J_vals->mat_data[2] = (theta_B-2*M_PI)*cos((v.mat_data[1]*(theta_B-2*M_PI)+2*M_PI)/2);
    J_vals->mat_data[3] = -(theta_B-2*M_PI)*cos(v.mat_data[1]*(theta_B-2*M_PI)+2*M_PI);
}

void f(rd_mat_t x, rd_mat_t y, rd_mat_t *f_xy) {
    rd_mat_shape(f_xy, x.rows, x.columns);
    for (MKL_INT i = 0; i < x.rows*x.columns; i++) {
        f_xy->mat_data[i] = 4 + (1 + pow(x.mat_data[i], 2) + pow(y.mat_data[i], 2))*(sin(2.5*M_PI*x.mat_data[i] - 0.5) + cos(2*M_PI*y.mat_data[i] - 0.5));
    }
}

int main() {
    M_p_C2_extra_param_t M_p_params = {0.4, 2*M_PI-0.4, 0};
    M_p_t C2_M_p = {(M_p_handle_t) M_p_C2, (void*) &M_p_params};
    J_C2_extra_param_t J_params = {0.4, 2*M_PI-0.4};
    J_t C2_J = {(J_handle_t) J_C2, (void*) &J_params};

    MKL_INT n_xi = 40;
    MKL_INT n_eta = 40;

    double f_XY_data[(n_xi+1)*(n_eta+1)];
    rd_mat_t f_XY = rd_mat_init(f_XY_data, n_xi+1, n_eta+1);

    q_patch_t C2_patch_test;
    q_patch_init(&C2_patch_test, C2_M_p, C2_J, 1e-13, 1e-13, n_xi, n_eta, 0.0, 1.0, 0.0, 1.0, &f_XY, NULL);
    
    double XI_data[q_patch_grid_num_el(&C2_patch_test)];
    double ETA_data[q_patch_grid_num_el(&C2_patch_test)];
    rd_mat_t XI = rd_mat_init_no_shape(XI_data);
    rd_mat_t ETA = rd_mat_init_no_shape(ETA_data);

    double X_data[q_patch_grid_num_el(&C2_patch_test)];
    double Y_data[q_patch_grid_num_el(&C2_patch_test)];
    rd_mat_t X = rd_mat_init_no_shape(X_data);
    rd_mat_t Y = rd_mat_init_no_shape(Y_data);

    q_patch_xi_eta_mesh(&C2_patch_test, &XI, &ETA);

    q_patch_xy_mesh(&C2_patch_test, &X, &Y);

    f(X, Y, C2_patch_test.f_XY);

    inverse_M_p_return_type_t inverse_return = q_patch_inverse_M_p(&C2_patch_test, 0.6, 0.4, NULL, NULL);
    printf("%f, %f, %d\n", inverse_return.xi, inverse_return.eta, inverse_return.converged);

    //reading continuation matrices
    double A_data[fc_A_numel(5, 27, 6)];
    double Q_data[fc_Q_numel(5)];
    rd_mat_t A = {A_data, 0, 0};
    rd_mat_t Q = {Q_data, 0, 0};

    read_fc_matrix(5, 27, 6, "fc_data/A_d5_C27.txt", "fc_data/Q_d5_C27.txt", &A, &Q);

    return 0;
}