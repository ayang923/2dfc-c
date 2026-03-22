#ifndef __NUM_LINALG_LIB__
#define __NUM_LINALG_LIB__

#include <mkl.h>
#include <stdlib.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

/** @brief Function pointer for a scalar-valued function of one variable. */
typedef double (*scalar_func_t) (double theta);

/** @brief Function pointer for a scalar-valued function of two variables. */
typedef double (*scalar_func_2D_t) (double x, double y);

/**
 * @brief Real double-precision matrix stored in column-major order.
 *
 * @param mat_data  Pointer to the underlying data array (column-major).
 * @param rows      Number of rows.
 * @param columns   Number of columns.
 */
typedef struct rd_mat {
    double *mat_data;
    MKL_INT rows;
    MKL_INT columns;
} rd_mat_t;

/**
 * @brief Integer matrix stored in column-major order.
 *
 * @param mat_data  Pointer to the underlying data array (column-major).
 * @param rows      Number of rows.
 * @param columns   Number of columns.
 */
typedef struct ri_mat {
    int *mat_data;
    MKL_INT rows;
    MKL_INT columns;
} ri_mat_t;

/**
 * @brief (row, column) subscript into a 2D matrix.
 *
 * @param i  Row index (0-based).
 * @param j  Column index (0-based).
 */
typedef struct sub {
    MKL_INT i;
    MKL_INT j;
} sub_t;

/**
 * @brief Converts a (row, column) subscript to a column-major linear index.
 *
 * @param rows     Number of rows in the matrix.
 * @param columns  Number of columns in the matrix.
 * @param sub      The (i, j) subscript.
 * @return         Linear index into the column-major data array.
 */
MKL_INT sub2ind(MKL_INT rows, MKL_INT columns, sub_t sub);

/**
 * @brief Converts a column-major linear index to a (row, column) subscript.
 *
 * @param rows     Number of rows in the matrix.
 * @param columns  Number of columns in the matrix.
 * @param idx      Linear index into the column-major data array.
 * @return         The corresponding (i, j) subscript.
 */
sub_t ind2sub(MKL_INT rows, MKL_INT columns, MKL_INT idx);

/**
 * @brief Fills mat with n evenly spaced values in [start, end] (inclusive).
 *        Equivalent to MATLAB's linspace.
 *
 * @param start     First value.
 * @param end       Last value.
 * @param n         Number of points.
 * @param mat_addr  Output column vector (must be pre-shaped as (n, 1)).
 */
void rd_linspace(double start, double end, MKL_INT n, rd_mat_t *mat_addr);

/**
 * @brief Creates 2D coordinate matrices from 1D coordinate vectors.
 *        Equivalent to MATLAB's meshgrid(x, y).
 *
 * @param x  Column vector of x-coordinates (length n_x).
 * @param y  Column vector of y-coordinates (length n_y).
 * @param X  Output (n_y × n_x) matrix where each row is a copy of x.
 * @param Y  Output (n_y × n_x) matrix where each column is a copy of y.
 */
void rd_meshgrid(rd_mat_t x, rd_mat_t y, rd_mat_t *X, rd_mat_t *Y);

/**
 * @brief Clamps an integer index mesh to [min_bound, max_bound] by shifting.
 *        If the mesh starts below min_bound, it is shifted up; if it ends above
 *        max_bound, it is shifted down.  The mesh length is preserved.
 *
 * @param mat        Integer column vector of indices to clamp (modified in place).
 * @param min_bound  Lower bound (inclusive).
 * @param max_bound  Upper bound (inclusive).
 */
void shift_idx_mesh(ri_mat_t *mat, int min_bound, int max_bound);

/**
 * @brief Creates an rd_mat_t wrapper around an existing data buffer with shape.
 *
 * @param mat_data_addr  Pointer to the data buffer.
 * @param rows           Number of rows.
 * @param columns        Number of columns.
 * @return               Initialized rd_mat_t.
 */
rd_mat_t rd_mat_init(double *mat_data_addr, MKL_INT rows, MKL_INT columns);

/**
 * @brief Creates an rd_mat_t wrapper with rows = columns = 0 (shape set later).
 *
 * @param mat_data_addr  Pointer to the data buffer.
 * @return               Initialized rd_mat_t with zero dimensions.
 */
rd_mat_t rd_mat_init_no_shape(double *mat_data_addr);

/**
 * @brief Creates an ri_mat_t wrapper around an existing integer data buffer.
 *
 * @param mat_data_addr  Pointer to the integer data buffer.
 * @param rows           Number of rows.
 * @param columns        Number of columns.
 * @return               Initialized ri_mat_t.
 */
ri_mat_t ri_mat_init(int *mat_data_addr, MKL_INT rows, MKL_INT columns);

/**
 * @brief Creates an ri_mat_t wrapper with rows = columns = 0 (shape set later).
 *
 * @param mat_data_addr  Pointer to the integer data buffer.
 * @return               Initialized ri_mat_t with zero dimensions.
 */
ri_mat_t ri_mat_init_no_shape(int *mat_data_addr);

/**
 * @brief Sets the rows and columns fields of an rd_mat_t without touching data.
 *
 * @param mat      Matrix whose shape is being set.
 * @param rows     New row count.
 * @param columns  New column count.
 */
void rd_mat_shape(rd_mat_t *mat, MKL_INT rows, MKL_INT columns);

/**
 * @brief Sets the rows and columns fields of an ri_mat_t without touching data.
 *
 * @param mat      Matrix whose shape is being set.
 * @param rows     New row count.
 * @param columns  New column count.
 */
void ri_mat_shape(ri_mat_t *mat, MKL_INT rows, MKL_INT columns);

/**
 * @brief Fills mat with the integer sequence start, start+step, ..., end.
 *
 * @param start      First value in the sequence.
 * @param step_size  Step between consecutive values (may be negative).
 * @param end        Last value in the sequence (inclusive).
 * @param mat_addr   Output column vector (must be pre-shaped with the correct length).
 */
void ri_range(int start, int step_size, int end, ri_mat_t *mat_addr);

/**
 * @brief Creates 2D coordinate matrices from 1D integer coordinate vectors.
 *        Equivalent to MATLAB's meshgrid(x, y) for integer arrays.
 *
 * @param x  Integer column vector of x-indices (length n_x).
 * @param y  Integer column vector of y-indices (length n_y).
 * @param X  Output (n_y × n_x) matrix where each row is a copy of x.
 * @param Y  Output (n_y × n_x) matrix where each column is a copy of y.
 */
void ri_meshgrid(ri_mat_t x, ri_mat_t y, ri_mat_t *X, ri_mat_t *Y);

/**
 * @brief Prints all elements of a double matrix to stdout (row-by-row, 16 digits).
 *
 * @param mat  Matrix to print.
 */
void print_matrix(rd_mat_t mat);

/**
 * @brief Prints all elements of an integer matrix to stdout (row-by-row).
 *
 * @param mat  Matrix to print.
 */
void ri_print_matrix(ri_mat_t mat);

/**
 * @brief Evaluates the barycentric Lagrange interpolating polynomial at x.
 *
 * Given n interpolation nodes ix and corresponding values iy, returns the
 * unique polynomial of degree ≤ n-1 evaluated at x.  If x coincides with
 * a node the exact node value is returned immediately.
 *
 * @param ix  Column vector of interpolation nodes (length n, assumed distinct).
 * @param iy  Column vector of function values at the nodes (length n).
 * @param x   Evaluation point.
 * @return    Interpolated value p(x).
 */
double barylag(rd_mat_t ix, rd_mat_t iy, double x);

#endif
