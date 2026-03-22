#ifndef __FC_LIB__
#define __FC_LIB__

#include <stdlib.h>

#include "num_linalg_lib.h"
#include "curve_seq_lib.h"

int read_fc_matrix(MKL_INT d, MKL_INT C, MKL_INT n_r, char* A_filename, char* Q_filename, rd_mat_t *A, rd_mat_t *Q);

MKL_INT fc_A_numel(MKL_INT d, MKL_INT C, MKL_INT n_r);

MKL_INT fc_Q_numel(MKL_INT d);

void fcont_gram_blend_S(rd_mat_t fx, MKL_INT d, rd_mat_t A, rd_mat_t Q, rd_mat_t *fcont);

#endif