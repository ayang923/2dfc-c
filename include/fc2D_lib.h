#ifndef __FC2D_LIB__
#define __FC2D_LIB__

#include <stdlib.h>

#include "num_linalg_lib.h"
#include "curve_seq_lib.h"

void FC2D(scalar_func_2D_t f, double h, curve_seq_t curve_seq, double eps_xi_eta, double eps_xy, MKL_INT d, MKL_INT C, MKL_INT n_r, rd_mat_t A, rd_mat_t Q, MKL_INT M, MKL_INT n_x_fft, MKL_INT n_y_fft);

void FC2D_heap(scalar_func_2D_t f, double h, curve_seq_t curve_seq, double eps_xi_eta, double eps_xy, MKL_INT d, MKL_INT C, MKL_INT n_r, rd_mat_t A, rd_mat_t Q, MKL_INT M, MKL_INT n_x_fft, MKL_INT n_y_fft);
#endif