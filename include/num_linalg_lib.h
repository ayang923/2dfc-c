#ifndef __NUM_LINALG_LIB__
#define __NUM_LINALG_LIB__

#include <mkl.h>
#include <stdlib.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

typedef double (*scalar_func_t) (double theta);

typedef double (*scalar_func_2D_t) (double x, double y);

/**
 * @brief Real, double matrix
 * 
 * @param mat_data: array that contains data organized column major
 * @param rows: number of rows
 * @param columns: number of columns
 */
typedef struct rd_mat {
    double *mat_data;
    MKL_INT rows;
    MKL_INT columns;
} rd_mat_t;

typedef struct ri_mat {
    int *mat_data;
    MKL_INT rows;
    MKL_INT columns;
} ri_mat_t;

/**
 * @brief subscript struct
 * 
 * @param i: row number
 * @param j: column number
 */
typedef struct sub {
    MKL_INT i;
    MKL_INT j;
} sub_t;

/**
 * @brief Converts subscript to index
 * 
 * @param rows: # rows in matrix
 * @param columns: # col in matrix
 * @param sub: subscript
 * @return MKL_INT: index
 */
MKL_INT sub2ind(MKL_INT rows, MKL_INT columns, sub_t sub);

/**
 * @brief Converts subscript to index
 * 
 * @param rows: # rows in matrix
 * @param columns: # col in matrix
 * @param idx: index
 * @return sub_t: subscript
 */
sub_t ind2sub(MKL_INT rows, MKL_INT columns, MKL_INT idx);

/**
 * @brief Implementation of matlab linspace using double
 * 
 * @param start 
 * @param end 
 * @param n 
 * @param mat_addr matrix we fill
 */
void rd_linspace(double start, double end, MKL_INT n, rd_mat_t *mat_addr);

void rd_meshgrid(rd_mat_t x, rd_mat_t y, rd_mat_t *X, rd_mat_t *Y);

void shift_idx_mesh(ri_mat_t *mat, int min_bound, int max_bound);

/**
 * @brief Wrapper function to create rd_mat_t object
 */
rd_mat_t rd_mat_init(double *mat_data_addr, MKL_INT rows, MKL_INT columns);

/**
 * @brief Wrapper function to create rd_mat_t object for rows and columns later
 */
rd_mat_t rd_mat_init_no_shape(double *mat_data_addr);

ri_mat_t ri_mat_init(int *mat_data_addr, MKL_INT rows, MKL_INT columns);

ri_mat_t ri_mat_init_no_shape(int *mat_data_addr);

void rd_mat_shape(rd_mat_t *mat, MKL_INT rows, MKL_INT columns);

void ri_mat_shape(ri_mat_t *mat, MKL_INT rows, MKL_INT columns);

void ri_range(int start, int step_size, int end, ri_mat_t *mat_addr);

void ri_meshgrid(ri_mat_t x, ri_mat_t y, ri_mat_t *X, ri_mat_t *Y);

void print_matrix(rd_mat_t mat);

void ri_print_matrix(ri_mat_t mat);

double barylag(rd_mat_t ix, rd_mat_t iy, double x);

#endif