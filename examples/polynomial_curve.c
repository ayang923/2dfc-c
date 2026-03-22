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

double f(double x, double y) {return -(pow(x+1, 2)+pow(y+1, 2))*sin(10*M_PI*x)*sin(10*M_PI*y);}

double curve_1_l_1(double theta) {return -theta+1;}

double curve_1_l_2(double theta) {return -theta+1;}

double curve_1_l_1_prime(double theta) {return -1;}

double curve_1_l_2_prime(double theta) {return -1;}

double curve_1_l_1_dprime(double theta) {return 0;}

double curve_1_l_2_dprime(double theta) {return 0;}

double curve_2_l_1(double theta) {return theta;}

double curve_2_l_2(double theta) {return pow(theta, 3);}

double curve_2_l_1_prime(double theta) {return 1;}

double curve_2_l_2_prime(double theta) {return 3*pow(theta, 2);}

double curve_2_l_1_dprime(double theta) {return 0;}

double curve_2_l_2_dprime(double theta) {return 6*theta;}



int main() {

    double h = 0.000125;
    double h_tan = 2*h;
    //reading continuation matrices
    MKL_INT d = 7;
    MKL_INT C = 27;
    MKL_INT n_r = 12;

    MKL_INT M = d+3;

    double A_data[fc_A_numel(d, C, n_r)];
    double Q_data[fc_Q_numel(d)];
    rd_mat_t A = rd_mat_init_no_shape(A_data);
    rd_mat_t Q = rd_mat_init_no_shape(Q_data);

    char A_fp[100];
    char Q_fp[100];
    sprintf(A_fp, "fc_data/A_d%d_C%d_r%d.txt", d, C, n_r);
    sprintf(Q_fp, "fc_data/Q_d%d_C%d_r%d.txt", d, C, n_r);
    read_fc_matrix(d, C, n_r, A_fp, Q_fp, &A, &Q);

    curve_seq_t curve_seq;
    curve_seq_init(&curve_seq);

    curve_t curve_1;
    curve_seq_add_curve(&curve_seq, &curve_1, (scalar_func_t) curve_1_l_1, (scalar_func_t) curve_1_l_2, (scalar_func_t) curve_1_l_1_prime, (scalar_func_t) curve_1_l_2_prime, (scalar_func_t) curve_1_l_1_dprime, (scalar_func_t) curve_1_l_2_dprime, 0, 1.0/3.0, 1.0/3.0, 2.0/3.0, 2.0/3.0, h_tan);

    curve_t curve_2;
    curve_seq_add_curve(&curve_seq, &curve_2, (scalar_func_t) curve_2_l_1, (scalar_func_t) curve_2_l_2, (scalar_func_t) curve_2_l_1_prime, (scalar_func_t) curve_2_l_2_prime, (scalar_func_t) curve_2_l_1_dprime, (scalar_func_t) curve_2_l_2_dprime, 0, 1.0/3.0, 1.0/3.0, 2.0/3.0, 2.0/3.0, h_tan);

    FC2D(f, h, curve_seq, 5e-15, 5e-15, d, C, n_r, A, Q, M);

    return 0;
}