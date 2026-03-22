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

double f(double x, double y) {
  return 4 + (1+pow(x, 2)+pow(y, 2))*(sin(10.5*M_PI*x-0.5)+cos(10*M_PI*y-0.5));
}

// double f(double x, double y) {
//   return pow(3*M_PI, 2) * sin(3*M_PI*x-1) * sin(3*M_PI*y-1);
// }

double l1_1(double theta) {
    return 2*sin(0.25*M_PI*theta);
}

double l2_1(double theta) {
    return -sin(0.5*M_PI*theta);
}

double l1p_1(double theta) {
    return 0.5*M_PI*cos(0.25*M_PI*theta);
}

double l2p_1(double theta) {
    return -0.5*M_PI*cos(0.5*M_PI*theta);
}

double l1dp_1(double theta) {
    return -2*pow(0.25*M_PI, 2)*sin(0.25*M_PI*theta);
}

double l2dp_1(double theta) {
    return pow(0.5*M_PI, 2)*sin(0.5*M_PI*theta);
}

double l1_2(double theta) {
    double x1_1 = l1_1(1);
    double x2_1 = 1;
    double x1_2 = l2_1(1);
    double x2_2 = 0;

    double n = (x2_2-x1_2) / sqrt(pow(x2_2-x1_2, 2)+pow(x2_1-x1_1, 2));
    double c = 0.5*(x1_1+x2_1) + 0.1*n;
    return pow(1-theta, 2)*x1_1 + 2*(1-theta)*theta*c + pow(theta, 2)*x2_1;
}

double l2_2(double theta) {
    double x1_1 = l1_1(1);
    double x2_1 = 1;
    double x1_2 = l2_1(1);
    double x2_2 = 0;

    double n = (x1_1-x2_1) / sqrt(pow(x2_2-x1_2, 2)+pow(x2_1-x1_1, 2));
    double c = 0.5*(x1_2+x2_2) + 0.1*n;
    return pow(1-theta, 2)*x1_2 + 2*(1-theta)*theta*c + pow(theta, 2)*x2_2;
}

double l1p_2(double theta) {
    double x1_1 = l1_1(1);
    double x2_1 = 1;
    double x1_2 = l2_1(1);
    double x2_2 = 0;

    double n = (x2_2-x1_2) / sqrt(pow(x2_2-x1_2, 2)+pow(x2_1-x1_1, 2));
    double c = 0.5*(x1_1+x2_1) + 0.1*n;

    return 2*((theta-1)*x1_1+(1-2*theta)*c+theta*x2_1);
}

double l2p_2(double theta) {
    double x1_1 = l1_1(1);
    double x2_1 = 1;
    double x1_2 = l2_1(1);
    double x2_2 = 0;

    double n = (x1_1-x2_1) / sqrt(pow(x2_2-x1_2, 2)+pow(x2_1-x1_1, 2));
    double c = 0.5*(x1_2+x2_2) + 0.1*n;

    return 2*((theta-1)*x1_2+(1-2*theta)*c+theta*x2_2);
}

double l1dp_2(double theta) {
    double x1_1 = l1_1(1);
    double x2_1 = 1;
    double x1_2 = l2_1(1);
    double x2_2 = 0;

    double n = (x2_2-x1_2) / sqrt(pow(x2_2-x1_2, 2)+pow(x2_1-x1_1, 2));
    double c = 0.5*(x1_1+x2_1) + 0.1*n;

    return 2*x1_1-4*c+2*x2_1;
}

double l2dp_2(double theta) {
    double x1_1 = l1_1(1);
    double x2_1 = 1;
    double x1_2 = l2_1(1);
    double x2_2 = 0;

    double n = (x1_1-x2_1) / sqrt(pow(x2_2-x1_2, 2)+pow(x2_1-x1_1, 2));
    double c = 0.5*(x1_2+x2_2) + 0.1*n;

    return 2*x1_2-4*c+2*x2_2;
}


double l1_3(double theta) {
    double x1_1 = 1;
    double x2_1 = 2*sin(M_PI*0.75);
    double x1_2 = 0;
    double x2_2 = -sin(2*M_PI*0.75);

    double n = (x2_2-x1_2) / sqrt(pow(x2_2-x1_2, 2)+pow(x2_1-x1_1, 2));
    double c = 0.5*(x1_1+x2_1) - 0.1*n;
    return pow(1-theta, 2)*x1_1 + 2*(1-theta)*theta*c + pow(theta, 2)*x2_1;
}

double l2_3(double theta) {
    double x1_1 = 1;
    double x2_1 = 2*sin(M_PI*0.75);
    double x1_2 = 0;
    double x2_2 = -sin(2*M_PI*0.75);

    double n = (x1_1-x2_1) / sqrt(pow(x2_2-x1_2, 2)+pow(x2_1-x1_1, 2));
    double c = 0.5*(x1_2+x2_2) - 0.1*n;
    return pow(1-theta, 2)*x1_2 + 2*(1-theta)*theta*c + pow(theta, 2)*x2_2;
}

double l1p_3(double theta) {
    double x1_1 = 1;
    double x2_1 = 2*sin(M_PI*0.75);
    double x1_2 = 0;
    double x2_2 = -sin(2*M_PI*0.75);

    double n = (x2_2-x1_2) / sqrt(pow(x2_2-x1_2, 2)+pow(x2_1-x1_1, 2));
    double c = 0.5*(x1_1+x2_1) - 0.1*n;

    return 2*((theta-1)*x1_1+(1-2*theta)*c+theta*x2_1);
}

double l2p_3(double theta) {
    double x1_1 = 1;
    double x2_1 = 2*sin(M_PI*0.75);
    double x1_2 = 0;
    double x2_2 = -sin(2*M_PI*0.75);

    double n = (x1_1-x2_1) / sqrt(pow(x2_2-x1_2, 2)+pow(x2_1-x1_1, 2));
    double c = 0.5*(x1_2+x2_2) - 0.1*n;

    return 2*((theta-1)*x1_2+(1-2*theta)*c+theta*x2_2);
}

double l1dp_3(double theta) {
    double x1_1 = 1;
    double x2_1 = 2*sin(M_PI*0.75);
    double x1_2 = 0;
    double x2_2 = -sin(2*M_PI*0.75);

    double n = (x2_2-x1_2) / sqrt(pow(x2_2-x1_2, 2)+pow(x2_1-x1_1, 2));
    double c = 0.5*(x1_1+x2_1) - 0.1*n;

    return 2*x1_1-4*c+2*x2_1;
}

double l2dp_3(double theta) {
    double x1_1 = 1;
    double x2_1 = 2*sin(M_PI*0.75);
    double x1_2 = 0;
    double x2_2 = -sin(2*M_PI*0.75);

    double n = (x1_1-x2_1) / sqrt(pow(x2_2-x1_2, 2)+pow(x2_1-x1_1, 2));
    double c = 0.5*(x1_2+x2_2) - 0.1*n;

    return 2*x1_2-4*c+2*x2_2;
}

double l1_4(double theta) {
    return 2*sin(M_PI*(0.25*theta+0.75));
}

double l2_4(double theta) {
    return -sin(2*M_PI*(0.25*theta+0.75));
}

double l1p_4(double theta) {
    return 0.5*M_PI*cos(M_PI*(0.25*theta+0.75));
}

double l2p_4(double theta) {
    return -0.5*M_PI*cos(2*M_PI*(0.25*theta+0.75));
}

double l1dp_4(double theta) {
    return -2*pow(0.25*M_PI, 2)*sin(M_PI*(0.25*theta+0.75));
}

double l2dp_4(double theta) {
    return pow(0.5*M_PI, 2)*sin(2*M_PI*(0.25*theta+0.75));
}

int main() {
    double h = 0.004;

    //reading continuation matrices
    MKL_INT d = 5;
    MKL_INT C = 27;
    MKL_INT n_r = 6;

    MKL_INT n_x_padded = 5832;
    MKL_INT n_y_padded = 8748;

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
   
    double n_frac_C_0;
    double n_frac_C_1;
    double n_frac_S_0;
    double n_frac_S_1;

    double h_tan;
    MKL_INT n_curve;
    
    curve_seq_t curve_seq;
    curve_seq_init(&curve_seq);

    h_tan = 2*h;
    n_curve = 0;

    n_frac_C_0 = 0.1;
    n_frac_C_1 = 0.2;
    n_frac_S_0 = 0.6;
    n_frac_S_1 = 0.7;
    curve_t curve_1;
    curve_seq_add_curve(&curve_seq, &curve_1, (scalar_func_t) l1_1, (scalar_func_t) l2_1, (scalar_func_t) l1p_1, (scalar_func_t) l2p_1, (scalar_func_t) l1dp_1, (scalar_func_t) l2dp_1, n_curve, n_frac_C_0, n_frac_C_1, n_frac_S_0, n_frac_S_1, h_tan);


    h_tan = 2*h;
    n_curve = 0;
    
    n_frac_C_0 = 0.3;
    n_frac_C_1 = 0.3;
    n_frac_S_0 = 0.7;
    n_frac_S_1 = 0.7;
    curve_t curve_2;
    curve_seq_add_curve(&curve_seq, &curve_2, (scalar_func_t) l1_2, (scalar_func_t) l2_2, (scalar_func_t) l1p_2, (scalar_func_t) l2p_2, (scalar_func_t) l1dp_2, (scalar_func_t) l2dp_2, n_curve, n_frac_C_0, n_frac_C_1, n_frac_S_0, n_frac_S_1, h_tan);

    h_tan = 2*h;
    n_curve = 0;

    n_frac_C_0 = 0.3;
    n_frac_C_1 = 0.3;
    n_frac_S_0 = 0.7;
    n_frac_S_1 = 0.7;
    curve_t curve_3;
    curve_seq_add_curve(&curve_seq, &curve_3, (scalar_func_t) l1_3, (scalar_func_t) l2_3, (scalar_func_t) l1p_3, (scalar_func_t) l2p_3, (scalar_func_t) l1dp_3, (scalar_func_t) l2dp_3, n_curve, n_frac_C_0, n_frac_C_1, n_frac_S_0, n_frac_S_1, h_tan);

    h_tan = 2*h;
    n_curve = 0;

    n_frac_C_0 = 0.3;
    n_frac_C_1 = 0.1;
    n_frac_S_0 = 0.7;
    n_frac_S_1 = 0.7;
    curve_t curve_4;
    curve_seq_add_curve(&curve_seq, &curve_4, (scalar_func_t) l1_4, (scalar_func_t) l2_4, (scalar_func_t) l1p_4, (scalar_func_t) l2p_4, (scalar_func_t) l1dp_4, (scalar_func_t) l2dp_4, n_curve, n_frac_C_0, n_frac_C_1, n_frac_S_0, n_frac_S_1, h_tan);

    
    FC2D(f, h, curve_seq, 1e-13, 1e-13, d, C, n_r, A, Q, M, n_x_padded, n_y_padded);

    return 0;
}
