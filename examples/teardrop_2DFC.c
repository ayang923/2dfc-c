#include <stdio.h>
#include <mkl.h>
#include <math.h>

#include "curve_seq_lib.h"
#include "s_patch_lib.h"
#include "q_patch_lib.h"
#include "fc_lib.h"
#include "r_cartesian_mesh_lib.h"
#include "time.h"
#include "fc2D_lib.h"

const double ALPH = 0.5;

double f(double x, double y) {
  return exp(0.5*(x*x+y*y))*(sin(10*M_PI*x)+cos(10*M_PI*y));
}

// double f(double x, double y) {
//   return pow(40*M_PI, 2) * sin(40*M_PI*x-1) * sin(40*M_PI*y-1);
// }

double l_1(double theta) {
    return 2*sin(theta*M_PI);
    
}

double l_2(double theta) {
    double bet = tan(ALPH*M_PI/2);
    return -bet * sin(theta*2*M_PI);
}

double l_1_prime(double theta) {
    return 2*M_PI*cos(theta*M_PI);
}

double l_2_prime(double theta) {
    double bet = tan(ALPH*M_PI/2);
    return -2*bet*M_PI*cos(theta*2*M_PI);
}

double l_1_dprime(double theta) {
    return -2*pow(M_PI, 2)*sin(theta*M_PI);
}

double l_2_dprime(double theta) {
    double bet = tan(ALPH*M_PI/2);
    return 4*bet*pow(M_PI, 2)*sin(theta*2*M_PI);
}

int main() {

    double h = 0.01;
    //reading continuation matrices
    MKL_INT d = 8;
    MKL_INT C = 27;
    MKL_INT n_r = 6;

    MKL_INT M = d+3;

    MKL_INT n_x_padded = 384;
    MKL_INT n_y_padded = 288;

    double A_data[fc_A_numel(d, C, n_r)];
    double Q_data[fc_Q_numel(d)];
    rd_mat_t A = rd_mat_init_no_shape(A_data);
    rd_mat_t Q = rd_mat_init_no_shape(Q_data);

    char A_fp[100];
    char Q_fp[100];
    sprintf(A_fp, "fc_data/A_d%d_C%d_r%d.txt", d, C, n_r);
    sprintf(Q_fp, "fc_data/Q_d%d_C%d_r%d.txt", d, C, n_r);
    read_fc_matrix(d, C, n_r, A_fp, Q_fp, &A, &Q);

    double h_norm = 1.5*h;
    double n_frac_c = 0.1;
    double n_frac_S = 0.6;
    MKL_INT n_curve = 0;
    
    curve_seq_t curve_seq;
    curve_seq_init(&curve_seq);
    
    curve_t curve_1;
    curve_seq_add_curve(&curve_seq, &curve_1, (scalar_func_t) l_1, (scalar_func_t) l_2, (scalar_func_t) l_1_prime, (scalar_func_t) l_2_prime, (scalar_func_t) l_1_dprime, (scalar_func_t) l_2_dprime, n_curve, n_frac_c, n_frac_c, n_frac_S, n_frac_S, h_norm);

    FC2D(f, h, curve_seq, 1e-13, 1e-13, d, C, n_r, A, Q, M, n_x_padded, n_y_padded);

    return 0;
}
