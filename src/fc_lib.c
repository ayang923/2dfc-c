#include <stdlib.h>
#include <stdio.h>

#include "fc_lib.h"
#include "num_linalg_lib.h"

int read_fc_matrix(MKL_INT d, MKL_INT C, MKL_INT n_r, char* A_filename, char* Q_filename, rd_mat_t *A, rd_mat_t *Q) {
    FILE *A_file = fopen(A_filename, "r");
    FILE *Q_file = fopen(Q_filename, "r");

    if (A_file == NULL || Q_file == NULL) {
        printf("Read failed");
        return 0;
    }

    Q->rows = d;
    Q->columns = d;

    A->rows = C*n_r;
    A->columns = d;

    char c;
    for (int i = 0; i < Q->rows; i++) {
        for (int j = 0; j < Q->columns; j++) {
            MKL_INT idx = sub2ind(Q->rows, Q->columns, (sub_t) {i, j});
            fscanf(Q_file, "%lf%c", Q->mat_data+idx, &c);
        }
    }

    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->columns; j++) {
            MKL_INT idx = sub2ind(A->rows, A->columns, (sub_t) {i, j});
            fscanf(A_file, "%lf%c", A->mat_data+idx, &c);
        }
    }

    return 1;
}

MKL_INT fc_A_numel(MKL_INT d, MKL_INT C, MKL_INT n_r) {
    return d*C*n_r;
}

MKL_INT fc_Q_numel(MKL_INT d) {
    return d*d;
}

void fcont_gram_blend_S(rd_mat_t fx, MKL_INT d, rd_mat_t A, rd_mat_t Q, rd_mat_t *fcont) {
    /*
     * Build the (d × n_cols) matching-point matrix from the first d rows of fx,
     * reversed in row order (flipud), so that the last sample is the first
     * 1D-BTZ matching mesh point (closest to the continuation interval).
     */
    double f_matching_data[d*fx.columns];

    fcont->rows = A.rows + 1;
    fcont->columns = fx.columns;

    for (int j = 0; j < fx.columns; j++) {
        for (int i = 0; i < d; i++) {
            MKL_INT fx_idx          = sub2ind(fx.rows, fx.columns, (sub_t) {i, j});
            MKL_INT f_matching_idx  = sub2ind(d, fx.columns, (sub_t) {d-1-i, j});
            f_matching_data[f_matching_idx] = fx.mat_data[fx_idx];

            /* Copy the boundary value (i == 0) as the last row of fcont. */
            if (i == 0) {
                MKL_INT fcont_idx = sub2ind(fcont->rows, fcont->columns, (sub_t) {fcont->rows-1, j});
                fcont->mat_data[fcont_idx] = fx.mat_data[fx_idx];
            }
        }
    }

    /* Project 1D-BTZ matching mesh onto the gram-blend basis: data_projection = Q^T * f_matching */
    double data_projection[d*fx.columns];
    double fcont_no_boundary[A.rows*fx.columns];
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                d, fx.columns, d,
                1.0, Q.mat_data, d, f_matching_data, d,
                0.0, data_projection, d);

    /* Evaluate the continuation: fcont_no_boundary = A * data_projection */
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                A.rows, fx.columns, d,
                1.0, A.mat_data, A.rows, data_projection, d,
                0.0, fcont_no_boundary, A.rows);

    /* Write continuation values in reverse row order into fcont (excluding the
     * last row, which was already filled with the boundary value above). */
    for (int j = 0; j < fcont->columns; j++) {
        for (int i = 0; i < fcont->rows-1; i++) {
            MKL_INT fcont_idx              = sub2ind(fcont->rows, fcont->columns, (sub_t) {i, j});
            MKL_INT fcont_no_boundary_idx  = sub2ind(A.rows, fx.columns, (sub_t) {fcont->rows-2-i, j});
            fcont->mat_data[fcont_idx] = fcont_no_boundary[fcont_no_boundary_idx];
        }
    }
}
