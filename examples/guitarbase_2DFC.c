// GUITARBASE_2DFC 2DFC example: guitar-body-shaped domain with 4 curves.
//
// Demonstrates the 2DFC algorithm on a multi-curve domain shaped like the base
// of a guitar body. The domain is formed by four connected curves:
//
//   Curve 1: bottom-right arc  - lower-right quarter of the teardrop curve
//   Curve 2: right Bezier arc  - quadratic Bezier connecting the right tip to the top-right
//   Curve 3: left Bezier arc   - quadratic Bezier connecting top-right to the upper-left
//   Curve 4: top-left arc      - upper-left quarter of the teardrop curve
//
// Curves 2 and 3 are quadratic Bezier curves that create the characteristic
// waist indentation of a guitar body.
//
// The test function is:
//   f(x,y) = 4 + (1 + x^2 + y^2) * (sin(10.5*pi*x - 0.5) + cos(pi*y - 0.5))

#include <stdio.h>
#include <mkl.h>
#include <math.h>

#include "curve_seq_lib.h"
#include "fc2D_lib.h"
#include "fc_lib.h"

double f(double x, double y) {
    return 4 + (1 + x*x + y*y) * (sin(10.5*M_PI*x - 0.5) + cos(M_PI*y - 0.5));
}

/* --- Curve 1: lower-right arc of teardrop (theta from 0 to 0.25) --- */

double l1_1(double theta) { return 2*sin(0.25*M_PI*theta); }
double l2_1(double theta) {
    double beta = tan(0.5*M_PI/2);
    return -beta * sin(0.5*M_PI*theta);
}
double l1p_1(double theta) { return 0.5*M_PI * cos(0.25*M_PI*theta); }
double l2p_1(double theta) {
    double beta = tan(0.5*M_PI/2);
    return -0.5*M_PI*beta * cos(0.5*M_PI*theta);
}
double l1dp_1(double theta) { return -2*pow(0.25*M_PI, 2) * sin(0.25*M_PI*theta); }
double l2dp_1(double theta) {
    double beta = tan(0.5*M_PI/2);
    return 4*beta*pow(0.25*M_PI, 2) * sin(0.5*M_PI*theta);
}

/* --- Curve 2: right Bezier arc connecting curve 1 end to (1, 0) ---
 * Quadratic Bezier with control point offset inward from the chord midpoint. */

static double c2_x1, c2_y1, c2_x2, c2_y2, c2_cx, c2_cy;

static void init_curve2(void) {
    c2_x1 = l1_1(1);  c2_y1 = l2_1(1);
    c2_x2 = 1.0;      c2_y2 = 0.0;
    double nx = -(c2_y2 - c2_y1);
    double ny =   c2_x2 - c2_x1;
    double len = sqrt(nx*nx + ny*ny);
    c2_cx = 0.5*(c2_x1 + c2_x2) + 0.1*(nx/len);
    c2_cy = 0.5*(c2_y1 + c2_y2) + 0.1*(ny/len);
}

double l1_2(double theta) { return (1-theta)*(1-theta)*c2_x1 + 2*(1-theta)*theta*c2_cx + theta*theta*c2_x2; }
double l2_2(double theta) { return (1-theta)*(1-theta)*c2_y1 + 2*(1-theta)*theta*c2_cy + theta*theta*c2_y2; }
double l1p_2(double theta) { return 2*((theta-1)*c2_x1 + (1-2*theta)*c2_cx + theta*c2_x2); }
double l2p_2(double theta) { return 2*((theta-1)*c2_y1 + (1-2*theta)*c2_cy + theta*c2_y2); }
double l1dp_2(double theta) { (void)theta; return 2*c2_x1 - 4*c2_cx + 2*c2_x2; }
double l2dp_2(double theta) { (void)theta; return 2*c2_y1 - 4*c2_cy + 2*c2_y2; }

/* --- Curve 3: left Bezier arc connecting (1, 0) to upper-left point on teardrop ---
 * Evaluate teardrop at theta=0.75 to get the upper-left endpoint. */

static double c3_x1, c3_y1, c3_x2, c3_y2, c3_cx, c3_cy;

static void init_curve3(void) {
    double beta = tan(0.5*M_PI/2);
    c3_x1 = 1.0;      c3_y1 = 0.0;
    c3_x2 = 2*sin(M_PI*0.75);
    c3_y2 = -beta * sin(2*M_PI*0.75);
    double nx = -(c3_y2 - c3_y1);
    double ny =   c3_x2 - c3_x1;
    double len = sqrt(nx*nx + ny*ny);
    c3_cx = 0.5*(c3_x1 + c3_x2) - 0.1*(nx/len);
    c3_cy = 0.5*(c3_y1 + c3_y2) - 0.1*(ny/len);
}

double l1_3(double theta) { return (1-theta)*(1-theta)*c3_x1 + 2*(1-theta)*theta*c3_cx + theta*theta*c3_x2; }
double l2_3(double theta) { return (1-theta)*(1-theta)*c3_y1 + 2*(1-theta)*theta*c3_cy + theta*theta*c3_y2; }
double l1p_3(double theta) { return 2*((theta-1)*c3_x1 + (1-2*theta)*c3_cx + theta*c3_x2); }
double l2p_3(double theta) { return 2*((theta-1)*c3_y1 + (1-2*theta)*c3_cy + theta*c3_y2); }
double l1dp_3(double theta) { (void)theta; return 2*c3_x1 - 4*c3_cx + 2*c3_x2; }
double l2dp_3(double theta) { (void)theta; return 2*c3_y1 - 4*c3_cy + 2*c3_y2; }

/* --- Curve 4: upper-left arc of teardrop (theta from 0.75 to 1) --- */

double l1_4(double theta) { return 2*sin(M_PI*(0.25*theta + 0.75)); }
double l2_4(double theta) {
    double beta = tan(0.5*M_PI/2);
    return -beta * sin(2*M_PI*(0.25*theta + 0.75));
}
double l1p_4(double theta) { return 0.5*M_PI * cos(M_PI*(0.25*theta + 0.75)); }
double l2p_4(double theta) {
    double beta = tan(0.5*M_PI/2);
    return -0.5*M_PI*beta * cos(2*M_PI*(0.25*theta + 0.75));
}
double l1dp_4(double theta) { return -2*pow(0.25*M_PI, 2) * sin(M_PI*(0.25*theta + 0.75)); }
double l2dp_4(double theta) {
    double beta = tan(0.5*M_PI/2);
    return 4*beta*pow(0.25*M_PI, 2) * sin(2*M_PI*(0.25*theta + 0.75));
}

int main() {
    /* Initialize Bezier control points before curve functions are used. */
    init_curve2();
    init_curve3();

    /* ---- parameters ---- */
    double  h          = 0.0005;
    MKL_INT d          = 6;
    MKL_INT n_x_padded = 3072;
    MKL_INT n_y_padded = 4374;
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

    double h_norm  = 2*h;
    MKL_INT n_curve = 0;

    curve_seq_t curve_seq;
    curve_seq_init(&curve_seq);

    /* Curve 1: lower-right arc */
    curve_t curve_1;
    curve_seq_add_curve(&curve_seq, &curve_1,
        (scalar_func_t) l1_1,  (scalar_func_t) l2_1,
        (scalar_func_t) l1p_1, (scalar_func_t) l2p_1,
        (scalar_func_t) l1dp_1,(scalar_func_t) l2dp_1,
        n_curve, 0.1, 0.2, 0.6, 0.7, h_norm);

    /* Curve 2: right Bezier arc */
    curve_t curve_2;
    curve_seq_add_curve(&curve_seq, &curve_2,
        (scalar_func_t) l1_2,  (scalar_func_t) l2_2,
        (scalar_func_t) l1p_2, (scalar_func_t) l2p_2,
        (scalar_func_t) l1dp_2,(scalar_func_t) l2dp_2,
        n_curve, 0.3, 0.3, 0.7, 0.7, h_norm);

    /* Curve 3: left Bezier arc */
    curve_t curve_3;
    curve_seq_add_curve(&curve_seq, &curve_3,
        (scalar_func_t) l1_3,  (scalar_func_t) l2_3,
        (scalar_func_t) l1p_3, (scalar_func_t) l2p_3,
        (scalar_func_t) l1dp_3,(scalar_func_t) l2dp_3,
        n_curve, 0.3, 0.3, 0.7, 0.7, h_norm);

    /* Curve 4: upper-left arc */
    curve_t curve_4;
    curve_seq_add_curve(&curve_seq, &curve_4,
        (scalar_func_t) l1_4,  (scalar_func_t) l2_4,
        (scalar_func_t) l1p_4, (scalar_func_t) l2p_4,
        (scalar_func_t) l1dp_4,(scalar_func_t) l2dp_4,
        n_curve, 0.3, 0.1, 0.7, 0.7, h_norm);

    FC2D(f, h, curve_seq, 1e-13, 1e-13, d, C, n_r, A, Q, M, n_x_padded, n_y_padded);

    return 0;
}
