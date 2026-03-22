#include <stdlib.h>
#include <stdio.h>
#include <mkl.h>
#include <assert.h>
#include <float.h>
#include <math.h>

#include "num_linalg_lib.h"

MKL_INT sub2ind(MKL_INT rows, MKL_INT columns, sub_t sub) {
    assert(sub.i < rows && sub.j < columns);
    return sub.j*rows + sub.i;
}

sub_t ind2sub(MKL_INT rows, MKL_INT columns, MKL_INT idx) {
    assert(idx < columns*rows);
    return (sub_t) {idx%rows, idx/rows};
}

void rd_linspace(double start, double end, MKL_INT n, rd_mat_t *mat_addr) {
    assert(mat_addr->rows == n && mat_addr->columns == 1 && start < end);

    double h = (end-start) / (n-1);
    for (MKL_INT i = 0; i < n; i++) {
        mat_addr->mat_data[i] = start + i*h;
    }
}

void shift_idx_mesh(ri_mat_t *mat, int min_bound, int max_bound) {
    if (mat->mat_data[0] < min_bound) {
        ri_range(min_bound, 1, min_bound+mat->rows-1, mat);
    }
    if (mat->mat_data[mat->rows-1] > max_bound) {
        ri_range(max_bound-mat->rows+1, 1, max_bound, mat);
    }
}

void rd_meshgrid(rd_mat_t x, rd_mat_t y, rd_mat_t *X, rd_mat_t *Y) {
    // assumes x and y are column vectors

    X->rows = y.rows;
    X->columns = x.rows;
    Y->rows = y.rows;
    Y->columns = x.rows;

    for (MKL_INT i = 0; i < y.rows; i++) {
        for (MKL_INT j = 0; j < x.rows; j++) {
            MKL_INT idx = sub2ind(y.rows, x.rows, (sub_t) {i, j});
            X->mat_data[idx] = x.mat_data[j];
            Y->mat_data[idx] = y.mat_data[i];
        }
    }
}

rd_mat_t rd_mat_init(double *mat_data_addr, MKL_INT rows, MKL_INT columns) {
    return (rd_mat_t) {mat_data_addr, rows, columns};
}

rd_mat_t rd_mat_init_no_shape(double *mat_data_addr) {
    return rd_mat_init(mat_data_addr, 0, 0);
}

ri_mat_t ri_mat_init(int *mat_data_addr, MKL_INT rows, MKL_INT columns) {
    return (ri_mat_t) {mat_data_addr, rows, columns};
}

ri_mat_t ri_mat_init_no_shape(int *mat_data_addr) {
    return ri_mat_init(mat_data_addr, 0, 0);
}

void rd_mat_shape(rd_mat_t *mat, MKL_INT rows, MKL_INT columns) {
    mat->rows = rows;
    mat->columns = columns;
}

void ri_mat_shape(ri_mat_t *mat, MKL_INT rows, MKL_INT columns) {
    mat->rows = rows;
    mat->columns = columns;
}

void ri_range(int start, int step_size, int end, ri_mat_t *mat_addr) {
    assert(mat_addr->rows == (end-start)/step_size+1);
    for (MKL_INT i = 0; i < mat_addr->rows; i++) {
        mat_addr->mat_data[i] = start + step_size*i;
    }
}

void ri_meshgrid(ri_mat_t x, ri_mat_t y, ri_mat_t *X, ri_mat_t *Y) {
    // assumes x and y are column vectors

    X->rows = y.rows;
    X->columns = x.rows;
    Y->rows = y.rows;
    Y->columns = x.rows;

    for (MKL_INT i = 0; i < y.rows; i++) {
        for (MKL_INT j = 0; j < x.rows; j++) {
            MKL_INT idx = sub2ind(y.rows, x.rows, (sub_t) {i, j});
            X->mat_data[idx] = x.mat_data[j];
            Y->mat_data[idx] = y.mat_data[i];
        }
    }
}

void ri_print_matrix(ri_mat_t mat) {
    for (MKL_INT i = 0; i < mat.rows; i++) {
        for (MKL_INT j = 0; j < mat.columns; j++) {
            MKL_INT idx = sub2ind(mat.rows, mat.columns, (sub_t) {i, j});
            printf("%d ", mat.mat_data[idx]);
        }
        printf("\n");
    }
}

double barylag(rd_mat_t ix, rd_mat_t iy, double x) {
    // assumes ix and iy are column vectors

    //computing weights
    MKL_INT n = ix.rows;
    double w[n];
    w[0] = 1;

    for (MKL_INT j = 1; j < n; j++) {
        w[j] = 1;
        for (MKL_INT k = 0; k < j; k++) {
            w[k] = (ix.mat_data[k]-ix.mat_data[j])*w[k];
            w[j] = (ix.mat_data[j]-ix.mat_data[k])*w[j];
        }
    }
    for (MKL_INT j = 0; j < n; j++) {
        w[j] = 1/w[j];
    }

    double w_x_distance[n];
    for (MKL_INT j = 0; j < n; j++) {
        if (fabs(ix.mat_data[j]-x) < DBL_EPSILON) {
            return iy.mat_data[j];
        }
        w_x_distance[j] = 1/(x-ix.mat_data[j]);
    }
    
    vdMul(n, w, w_x_distance, w_x_distance);

    double f_w_x_distance[n];
    vdMul(n, w_x_distance, iy.mat_data, f_w_x_distance);

    double one = 1;
    return cblas_ddot(n, f_w_x_distance, 1, &one, 0) / cblas_ddot(n, w_x_distance, 1, &one, 0);
}

void print_matrix(rd_mat_t mat) {
    for (MKL_INT i = 0; i < mat.rows; i++) {
        for (MKL_INT j = 0; j < mat.columns; j++) {
            MKL_INT idx = sub2ind(mat.rows, mat.columns, (sub_t) {i, j});
            printf("%.16f ", mat.mat_data[idx]);
        }
        printf("\n");
    }
}


