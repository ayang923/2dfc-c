// TEARDROP_SHARP_2DFC 2DFC example: extremely sharp teardrop domain.
//
// Same teardrop curve as teardrop_2DFC.c but with alpha = 0.01, giving an
// extremely narrow tip (beta ~ 0.016). The near-cusp geometry requires a
// very fine mesh (h = 2.5e-5) and a small corner patch fraction.
//
//   l_1(theta) = 2 * sin(pi*theta)
//   l_2(theta) = -beta * sin(2*pi*theta),  beta = tan(alpha*pi/2)
//
// The test function is:
//   f(x,y) = -((x+1)^2 + (y+1)^2) * sin(pi*(x-0.1)) * cos(pi*y)

#include <stdio.h>
#include <mkl.h>
#include <math.h>

#include "curve_seq_lib.h"
#include "fc2D_lib.h"
#include "fc_lib.h"

const double ALPH = 0.01;

double f(double x, double y) {
    return -((x+1)*(x+1) + (y+1)*(y+1)) * sin(M_PI*(x-0.1)) * cos(M_PI*y);
}

double l_1(double theta) {
    return 2*sin(theta*M_PI);
}

double l_2(double theta) {
    double bet = tan(ALPH*M_PI/2);
    return -bet * sin(theta*2*M_PI);
}

double l_1_prime(double theta) {
    return 2*M_PI * cos(theta*M_PI);
}

double l_2_prime(double theta) {
    double bet = tan(ALPH*M_PI/2);
    return -2*bet*M_PI * cos(theta*2*M_PI);
}

double l_1_dprime(double theta) {
    return -2*pow(M_PI, 2) * sin(theta*M_PI);
}

double l_2_dprime(double theta) {
    double bet = tan(ALPH*M_PI/2);
    return 4*bet*pow(M_PI, 2) * sin(theta*2*M_PI);
}

int main() {
    /* ---- parameters ---- */
    double  h          = 0.000025;
    MKL_INT d          = 7;
    MKL_INT n_x_padded = 82944;
    MKL_INT n_y_padded = 1458;
    /* -------------------- */

    MKL_INT C   = 27;
    MKL_INT n_r = 6;
    MKL_INT M   = d + 3;

    double A_data[fc_A_numel(d, C, n_r)];
    double Q_data[fc_Q_numel(d)];
    rd_mat_t A = rd_mat_init_no_shape(A_data);
    rd_mat_t Q = rd_mat_init_no_shape(Q_data);

    char A_fp[100], Q_fp[100];
    sprintf(A_fp, "fc_data/A_d%d_C%d_r%d.txt", d, C, n_r);
    sprintf(Q_fp, "fc_data/Q_d%d_C%d_r%d.txt", d, C, n_r);
    read_fc_matrix(d, C, n_r, A_fp, Q_fp, &A, &Q);

    double h_norm   = 1.5*h;
    double h_tan    = 16*h_norm;
    double n_frac_C = 0.025;
    double n_frac_S = 0.85;
    MKL_INT n_curve = (MKL_INT) ceil(4.0 / h_tan);

    curve_seq_t curve_seq;
    curve_seq_init(&curve_seq);

    curve_t curve_1;
    curve_seq_add_curve(&curve_seq, &curve_1,
        (scalar_func_t) l_1,       (scalar_func_t) l_2,
        (scalar_func_t) l_1_prime, (scalar_func_t) l_2_prime,
        (scalar_func_t) l_1_dprime,(scalar_func_t) l_2_dprime,
        n_curve, n_frac_C, n_frac_C, n_frac_S, n_frac_S, h_norm);

    FC2D_heap(f, h, curve_seq, 1e-13, 1e-13, d, C, n_r, A, Q, M, n_x_padded, n_y_padded);

    return 0;
}
