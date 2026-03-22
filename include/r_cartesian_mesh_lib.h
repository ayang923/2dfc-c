#ifndef __R_CARTESIAN_MESH_LIB__
#define __R_CARTESIAN_MESH_LIB__

#include "num_linalg_lib.h"
#include <mkl.h>
#include <string.h>

#include "q_patch_lib.h"


MKL_INT inpolygon_mesh(rd_mat_t R_X, rd_mat_t R_Y, rd_mat_t boundary_X, rd_mat_t boundary_Y, ri_mat_t *in_msk);

typedef struct r_cartesian_mesh_obj {
    double x_start;
    double x_end;
    double y_start;
    double y_end;
    double h;
    MKL_INT n_x;
    MKL_INT n_y;

    rd_mat_t *R_X;
    rd_mat_t *R_Y;

    ri_mat_t *in_interior;
    rd_mat_t *f_R;
} r_cartesian_mesh_obj_t;

MKL_INT r_cartesian_n_total(double x_start, double x_end, double y_start, double y_end, double h);

void r_cartesian_mesh_init(r_cartesian_mesh_obj_t *r_cartesian_mesh_obj, double x_start, double x_end, double y_start, double y_end, double h, rd_mat_t boundary_X, rd_mat_t boundary_Y, rd_mat_t *R_X, rd_mat_t *R_Y, ri_mat_t *in_interior, rd_mat_t *f_R);

void r_cartesian_mesh_interpolate_patch(r_cartesian_mesh_obj_t *r_cartesian_mesh_obj, q_patch_t *q_patch, MKL_INT M);

void r_cartesian_mesh_interpolate_patch_heap(r_cartesian_mesh_obj_t *r_cartesian_mesh_obj, q_patch_t *q_patch, MKL_INT M);

void r_cartesian_mesh_fill_interior(r_cartesian_mesh_obj_t *r_cartesian_mesh_obj, scalar_func_2D_t f);

double r_cartesian_mesh_compute_fc_error(r_cartesian_mesh_obj_t *r_cartesian_mesh_obj, scalar_func_2D_t f, MKL_INT rho_err, rd_mat_t boundary_X, rd_mat_t boundary_Y, MKL_INT n_x_fft, MKL_INT n_y_fft);

double r_cartesian_mesh_compute_fc_error_heap(r_cartesian_mesh_obj_t *r_cartesian_mesh_obj, scalar_func_2D_t f, MKL_INT rho_err, rd_mat_t boundary_X, rd_mat_t boundary_Y, MKL_INT n_x_fft, MKL_INT n_y_fft);

#endif