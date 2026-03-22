#ifndef __FC_LIB__
#define __FC_LIB__

#include <stdlib.h>

#include "num_linalg_lib.h"
#include "curve_seq_lib.h"

/**
 * @brief Reads precomputed FC matrices A and Q from text files and populates them.
 *
 * Expects files in the row-by-row format produced by the MATLAB FC-Gram toolbox.
 * A has shape (C*n_r × d) and Q has shape (d × d).
 *
 * @param d           1D-BTZ matching mesh size.
 * @param C           Number of continuation points.
 * @param n_r         Refinement factor.
 * @param A_filename  Path to the text file containing the A matrix.
 * @param Q_filename  Path to the text file containing the Q matrix.
 * @param A           Output: A matrix (data must be pre-allocated to fc_A_numel(d,C,n_r)).
 * @param Q           Output: Q matrix (data must be pre-allocated to fc_Q_numel(d)).
 * @return            1 on success, 0 if either file could not be opened.
 */
int read_fc_matrix(MKL_INT d, MKL_INT C, MKL_INT n_r, char* A_filename, char* Q_filename, rd_mat_t *A, rd_mat_t *Q);

/**
 * @brief Returns the number of elements in the FC continuation matrix A.
 *
 * @param d    1D-BTZ matching mesh size.
 * @param C    Number of continuation points.
 * @param n_r  Refinement factor.
 * @return     d * C * n_r.
 */
MKL_INT fc_A_numel(MKL_INT d, MKL_INT C, MKL_INT n_r);

/**
 * @brief Returns the number of elements in the FC orthogonalization matrix Q.
 *
 * @param d  1D-BTZ matching mesh size.
 * @return   d * d.
 */
MKL_INT fc_Q_numel(MKL_INT d);

/**
 * @brief Computes 1D Fourier Continuations for all columns of fx using the gram-blend method.
 *
 * For each column of fx (of length d), produces a smooth periodic extension of
 * length C*n_r+1 using the precomputed gram-blend matrices A and Q:
 *   1. The last d rows of fx are reversed to form a matching-point vector.
 *   2. The matching points are projected via Q^T to a coefficient vector.
 *   3. The coefficient vector is mapped to continuation values via A.
 *   4. The boundary value (fx[0]) is appended as the final row of fcont.
 *
 * @param fx     Input matrix of shape (n × n_cols); only the first d rows are used.
 * @param d      1D-BTZ matching mesh size.
 * @param A      FC continuation matrix of shape (C*n_r × d).
 * @param Q      FC gram-blend matrix of shape (d × d).
 * @param fcont  Output matrix of shape (C*n_r+1 × n_cols); must point to pre-allocated data.
 */
void fcont_gram_blend_S(rd_mat_t fx, MKL_INT d, rd_mat_t A, rd_mat_t Q, rd_mat_t *fcont);

#endif
