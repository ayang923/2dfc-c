#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#include "mkl_dfti.h"
#include "time.h"


#include "r_cartesian_mesh_lib.h"
#include "num_linalg_lib.h"
#include "q_patch_lib.h"

double round_end_bound(double start_bound, double end_bound, double h) {
    return ceil((end_bound-start_bound)/h)*h + start_bound;
}

MKL_INT n_1D(double start_bound, double end_bound, double h) {
    return round((round_end_bound(start_bound, end_bound, h)-start_bound)/h) + 1;
}

MKL_INT r_cartesian_n_total(double x_start, double x_end, double y_start, double y_end, double h) {
    return n_1D(x_start, x_end, h) * n_1D(y_start, y_end, h);
}

MKL_INT inpolygon_mesh(rd_mat_t R_X, rd_mat_t R_Y, rd_mat_t boundary_X, rd_mat_t boundary_Y, ri_mat_t *in_msk);

void r_cartesian_mesh_init(r_cartesian_mesh_obj_t *r_cartesian_mesh_obj, double x_start, double x_end, double y_start, double y_end, double h, rd_mat_t boundary_X, rd_mat_t boundary_Y, rd_mat_t *R_X, rd_mat_t *R_Y, ri_mat_t *in_interior, rd_mat_t *f_R) {
    r_cartesian_mesh_obj->x_start = x_start;
    r_cartesian_mesh_obj->y_start = y_start;

    r_cartesian_mesh_obj->x_end = round_end_bound(x_start, x_end, h);
    r_cartesian_mesh_obj->y_end = round_end_bound(y_start, y_end, h);

    r_cartesian_mesh_obj->h = h;
    
    r_cartesian_mesh_obj->n_x = n_1D(x_start, x_end, h);
    r_cartesian_mesh_obj->n_y = n_1D(y_start, y_end, h);

    r_cartesian_mesh_obj->R_X = R_X;
    r_cartesian_mesh_obj->R_Y = R_Y;
    r_cartesian_mesh_obj->in_interior = in_interior;

    rd_mat_shape(f_R, r_cartesian_mesh_obj->n_y, r_cartesian_mesh_obj->n_x);
    memset(f_R->mat_data, 0, r_cartesian_mesh_obj->n_y*r_cartesian_mesh_obj->n_x*sizeof(double));
    r_cartesian_mesh_obj->f_R = f_R;

    double x_mesh_data[r_cartesian_mesh_obj->n_x];
    double y_mesh_data[r_cartesian_mesh_obj->n_y];
    rd_mat_t x_mesh = rd_mat_init(x_mesh_data, r_cartesian_mesh_obj->n_x, 1);
    rd_mat_t y_mesh = rd_mat_init(y_mesh_data, r_cartesian_mesh_obj->n_y, 1);
    rd_linspace(r_cartesian_mesh_obj->x_start, r_cartesian_mesh_obj->x_end, r_cartesian_mesh_obj->n_x, &x_mesh);
    rd_linspace(r_cartesian_mesh_obj->y_start, r_cartesian_mesh_obj->y_end, r_cartesian_mesh_obj->n_y, &y_mesh);

    rd_meshgrid(x_mesh, y_mesh, R_X, R_Y);

    inpolygon_mesh(*R_X, *R_Y, boundary_X, boundary_Y, in_interior);
}

void r_cartesian_mesh_interpolate_patch(r_cartesian_mesh_obj_t *r_cartesian_mesh_obj, q_patch_t *q_patch, MKL_INT M) {
    double bound_X_data[q_patch_boundary_mesh_num_el(q_patch)];
    double bound_Y_data[q_patch_boundary_mesh_num_el(q_patch)];
    rd_mat_t bound_X = rd_mat_init_no_shape(bound_X_data);
    rd_mat_t bound_Y = rd_mat_init_no_shape(bound_Y_data);

    q_patch_boundary_mesh_xy(q_patch, false, &bound_X, &bound_Y);
    
    MKL_INT in_patch_data[r_cartesian_mesh_obj->n_x * r_cartesian_mesh_obj->n_y];
    ri_mat_t in_patch = ri_mat_init_no_shape(in_patch_data);
    MKL_INT n_in_patch = inpolygon_mesh(*(r_cartesian_mesh_obj->R_X), *(r_cartesian_mesh_obj->R_Y), bound_X, bound_Y, &in_patch);

    for (int i = 0; i < r_cartesian_mesh_obj->n_x * r_cartesian_mesh_obj->n_y; i++) {
        if (r_cartesian_mesh_obj->in_interior->mat_data[i] && in_patch_data[i]) {
            n_in_patch -= 1;
        }
        in_patch_data[i] = in_patch_data[i] && !(r_cartesian_mesh_obj->in_interior->mat_data[i]);        
    }

    MKL_INT r_patch_idxs_data[n_in_patch];
    ri_mat_t r_patch_idxs = ri_mat_init(r_patch_idxs_data, n_in_patch, 1);
    MKL_INT curr_idx = 0;
    for (int i = 0; i < r_cartesian_mesh_obj->n_x * r_cartesian_mesh_obj->n_y; i++) {
        if (in_patch.mat_data[i]) {
            r_patch_idxs.mat_data[curr_idx] = i;
            in_patch.mat_data[i] = curr_idx+1;
            curr_idx += 1;
        }
    }

    MKL_INT n_patch_grid = q_patch_grid_num_el(q_patch);
    double patch_XI_data[n_patch_grid];
    double patch_ETA_data[n_patch_grid];
    rd_mat_t patch_XI = rd_mat_init_no_shape(patch_XI_data);
    rd_mat_t patch_ETA = rd_mat_init_no_shape(patch_ETA_data);
    q_patch_xi_eta_mesh(q_patch, &patch_XI, &patch_ETA);

    double patch_X_data[n_patch_grid];
    double patch_Y_data[n_patch_grid];
    rd_mat_t patch_X = rd_mat_init_no_shape(patch_X_data);
    rd_mat_t patch_Y = rd_mat_init_no_shape(patch_Y_data);
    q_patch_convert_to_XY(q_patch, patch_XI, patch_ETA, &patch_X, &patch_Y);

    MKL_INT floor_X_j[n_patch_grid];
    MKL_INT ceil_X_j[n_patch_grid];
    MKL_INT floor_Y_j[n_patch_grid];
    MKL_INT ceil_Y_j[n_patch_grid];

    for (int i = 0; i < n_patch_grid; i++) {
        floor_X_j[i] = (MKL_INT) floor((patch_X.mat_data[i] - r_cartesian_mesh_obj->x_start)/r_cartesian_mesh_obj->h);
        ceil_X_j[i] = (MKL_INT) ceil((patch_X.mat_data[i] - r_cartesian_mesh_obj->x_start)/r_cartesian_mesh_obj->h);
        floor_Y_j[i] = (MKL_INT) floor((patch_Y.mat_data[i] - r_cartesian_mesh_obj->y_start)/r_cartesian_mesh_obj->h);
        ceil_Y_j[i] = (MKL_INT) ceil((patch_Y.mat_data[i] - r_cartesian_mesh_obj->y_start)/r_cartesian_mesh_obj->h);
    }

    double P_xi[n_in_patch];
    double P_eta[n_in_patch];

    for (int i = 0; i < n_in_patch; i++) {
        P_xi[i] = NAN;
        P_eta[i] = NAN;
    }

    for (int i = 0; i < n_patch_grid; i++) {
        double neighbors_X[4] = {floor_X_j[i], floor_X_j[i], ceil_X_j[i], ceil_X_j[i]};
        double neighbors_Y[4] = {floor_Y_j[i], ceil_Y_j[i], floor_Y_j[i], ceil_Y_j[i]};

        for (int j = 0; j < 4; j++) {
            double neighbor_X = neighbors_X[j];
            double neighbor_Y = neighbors_Y[j];

            if (neighbor_X > r_cartesian_mesh_obj->n_x-1 || neighbor_X < 0 || neighbor_Y > r_cartesian_mesh_obj->n_y-1 || neighbor_Y < 0) {
                continue;
            }
            MKL_INT patch_idx = sub2ind(r_cartesian_mesh_obj->n_y, r_cartesian_mesh_obj->n_x, (sub_t) {neighbor_Y, neighbor_X});
            
            if(in_patch.mat_data[patch_idx] != 0 && isnan(P_xi[in_patch.mat_data[patch_idx]-1])) {
                double neighbor_X_coord = neighbor_X*r_cartesian_mesh_obj->h+r_cartesian_mesh_obj->x_start;
                double neighbor_Y_coord = neighbor_Y*r_cartesian_mesh_obj->h+r_cartesian_mesh_obj->y_start;
                rd_mat_t initial_guess_xi = rd_mat_init(patch_XI_data+i, 1, 1);
                rd_mat_t initial_guess_eta = rd_mat_init(patch_ETA_data+i, 1, 1);
                inverse_M_p_return_type_t xi_eta = q_patch_inverse_M_p(q_patch, neighbor_X_coord, neighbor_Y_coord, &initial_guess_xi, &initial_guess_eta);

                if (xi_eta.converged) {
                    P_xi[in_patch.mat_data[patch_idx]-1] = xi_eta.xi;
                    P_eta[in_patch.mat_data[patch_idx]-1] = xi_eta.eta;
                }
                else {
                    printf("Nonconvergence in interpolation!!!");
                }
            }
        }
    }

    MKL_INT nan_count;
    bool first_iter = true;
    while (true) {
        if (!first_iter && nan_count == 0) {
            break;
        }
        nan_count = 0;
        for (int i = 0; i < n_in_patch; i++) {
            if (isnan(P_xi[i])) {
                bool is_touched = false;
                sub_t idx = ind2sub(r_cartesian_mesh_obj->n_y, r_cartesian_mesh_obj->n_x, r_patch_idxs_data[i]);

                MKL_INT neighbor_shift_x[8] = {1,-1, 0, 0, 1, 1, -1, -1};
                MKL_INT neighbor_shift_y[8] = {0, 0, -1, 1, 1, -1, 1, -1};
                for (int j = 0; j < 8; j++) {
                    MKL_INT neighbor_i = idx.i + neighbor_shift_y[j];
                    MKL_INT neighbor_j = idx.j + neighbor_shift_x[j];
                    
                    if (neighbor_j > r_cartesian_mesh_obj->n_x-1 || neighbor_j < 0 || neighbor_i > r_cartesian_mesh_obj->n_y-1 || neighbor_i < 0) {
                        continue;
                    }

                    MKL_INT neighbor = sub2ind(r_cartesian_mesh_obj->n_y, r_cartesian_mesh_obj->n_x, (sub_t) {neighbor_i, neighbor_j});
                    if (in_patch.mat_data[neighbor] != 0 && !isnan(P_xi[in_patch.mat_data[neighbor]-1])) {
                        double idx_x_coord = idx.j*r_cartesian_mesh_obj->h + r_cartesian_mesh_obj->x_start;
                        double idx_y_coord = idx.i*r_cartesian_mesh_obj->h + r_cartesian_mesh_obj->y_start;
                        rd_mat_t initial_guess_xi = rd_mat_init(P_xi + in_patch.mat_data[neighbor] - 1, 1, 1);
                        rd_mat_t initial_guess_eta = rd_mat_init(P_eta + in_patch.mat_data[neighbor] - 1, 1, 1);
                        inverse_M_p_return_type_t xi_eta = q_patch_inverse_M_p(q_patch, idx_x_coord, idx_y_coord, &initial_guess_xi, &initial_guess_eta);

                        if (xi_eta.converged) {
                            P_xi[i] = xi_eta.xi;
                            P_eta[i] = xi_eta.eta;
                            is_touched = true;
                        }
                        else {
                            printf("Nonconvergence in interpolation!!!");
                        }
                    }
                }
                if (!is_touched) {
                    nan_count += 1;
                }
            }
        }
        first_iter = false;
    }

    double f_R_patch[n_in_patch];
    for (int i = 0; i < n_in_patch; i++) {
        double xi_point = P_xi[in_patch.mat_data[r_patch_idxs.mat_data[i]]-1];
        double eta_point = P_eta[in_patch.mat_data[r_patch_idxs.mat_data[i]]-1];

        locally_compute_return_type_t f_locally_compute = q_patch_locally_compute(q_patch, xi_point, eta_point, M);
        if(f_locally_compute.in_range) {
            f_R_patch[i] = f_locally_compute.f_xy;	      
        }
	else {
            printf("WARNING: interpolating point not in patch\n");
	}
    }
    
    for (int i = 0; i < n_in_patch; i++) {
        r_cartesian_mesh_obj->f_R->mat_data[r_patch_idxs.mat_data[i]] += f_R_patch[i];
    }
}

void r_cartesian_mesh_interpolate_patch_heap(r_cartesian_mesh_obj_t *r_cartesian_mesh_obj, q_patch_t *q_patch, MKL_INT M) {
    double *bound_X_data = malloc(q_patch_boundary_mesh_num_el(q_patch) * sizeof(double));
    double *bound_Y_data = malloc(q_patch_boundary_mesh_num_el(q_patch) * sizeof(double));
    rd_mat_t bound_X = rd_mat_init_no_shape(bound_X_data);
    rd_mat_t bound_Y = rd_mat_init_no_shape(bound_Y_data);

    q_patch_boundary_mesh_xy(q_patch, false, &bound_X, &bound_Y);
    
    MKL_INT *in_patch_data = malloc(r_cartesian_mesh_obj->n_x * r_cartesian_mesh_obj->n_y * sizeof(MKL_INT));
    ri_mat_t in_patch = ri_mat_init_no_shape(in_patch_data);
    MKL_INT n_in_patch = inpolygon_mesh(*(r_cartesian_mesh_obj->R_X), *(r_cartesian_mesh_obj->R_Y), bound_X, bound_Y, &in_patch);

    for (int i = 0; i < r_cartesian_mesh_obj->n_x * r_cartesian_mesh_obj->n_y; i++) {
        if (r_cartesian_mesh_obj->in_interior->mat_data[i] && in_patch_data[i]) {
            n_in_patch -= 1;
        }
        in_patch_data[i] = in_patch_data[i] && !(r_cartesian_mesh_obj->in_interior->mat_data[i]);        
    }

    MKL_INT *r_patch_idxs_data = malloc(n_in_patch * sizeof(MKL_INT));
    ri_mat_t r_patch_idxs = ri_mat_init(r_patch_idxs_data, n_in_patch, 1);
    MKL_INT curr_idx = 0;
    for (int i = 0; i < r_cartesian_mesh_obj->n_x * r_cartesian_mesh_obj->n_y; i++) {
        if (in_patch.mat_data[i]) {
            r_patch_idxs.mat_data[curr_idx] = i;
            in_patch.mat_data[i] = curr_idx+1;
            curr_idx += 1;
        }
    }

    MKL_INT n_patch_grid = q_patch_grid_num_el(q_patch);
    double *patch_XI_data = malloc(n_patch_grid * sizeof(double));
    double *patch_ETA_data = malloc(n_patch_grid * sizeof(double));
    rd_mat_t patch_XI = rd_mat_init_no_shape(patch_XI_data);
    rd_mat_t patch_ETA = rd_mat_init_no_shape(patch_ETA_data);
    q_patch_xi_eta_mesh(q_patch, &patch_XI, &patch_ETA);

    double *patch_X_data = malloc(n_patch_grid * sizeof(double));
    double *patch_Y_data = malloc(n_patch_grid * sizeof(double));
    rd_mat_t patch_X = rd_mat_init_no_shape(patch_X_data);
    rd_mat_t patch_Y = rd_mat_init_no_shape(patch_Y_data);
    q_patch_convert_to_XY(q_patch, patch_XI, patch_ETA, &patch_X, &patch_Y);

    MKL_INT *floor_X_j = malloc(n_patch_grid * sizeof(MKL_INT));
    MKL_INT *ceil_X_j = malloc(n_patch_grid * sizeof(MKL_INT));
    MKL_INT *floor_Y_j = malloc(n_patch_grid * sizeof(MKL_INT));
    MKL_INT *ceil_Y_j = malloc(n_patch_grid * sizeof(MKL_INT));

    for (int i = 0; i < n_patch_grid; i++) {
        floor_X_j[i] = (MKL_INT) floor((patch_X.mat_data[i] - r_cartesian_mesh_obj->x_start)/r_cartesian_mesh_obj->h);
        ceil_X_j[i] = (MKL_INT) ceil((patch_X.mat_data[i] - r_cartesian_mesh_obj->x_start)/r_cartesian_mesh_obj->h);
        floor_Y_j[i] = (MKL_INT) floor((patch_Y.mat_data[i] - r_cartesian_mesh_obj->y_start)/r_cartesian_mesh_obj->h);
        ceil_Y_j[i] = (MKL_INT) ceil((patch_Y.mat_data[i] - r_cartesian_mesh_obj->y_start)/r_cartesian_mesh_obj->h);
    }

    double *P_xi = malloc(n_in_patch * sizeof(double));
    double *P_eta = malloc(n_in_patch * sizeof(double));

    for (int i = 0; i < n_in_patch; i++) {
        P_xi[i] = NAN;
        P_eta[i] = NAN;
    }

    for (int i = 0; i < n_patch_grid; i++) {
        double neighbors_X[4] = {floor_X_j[i], floor_X_j[i], ceil_X_j[i], ceil_X_j[i]};
        double neighbors_Y[4] = {floor_Y_j[i], ceil_Y_j[i], floor_Y_j[i], ceil_Y_j[i]};

        for (int j = 0; j < 4; j++) {
            double neighbor_X = neighbors_X[j];
            double neighbor_Y = neighbors_Y[j];

            if (neighbor_X > r_cartesian_mesh_obj->n_x-1 || neighbor_X < 0 || neighbor_Y > r_cartesian_mesh_obj->n_y-1 || neighbor_Y < 0) {
                continue;
            }
            MKL_INT patch_idx = sub2ind(r_cartesian_mesh_obj->n_y, r_cartesian_mesh_obj->n_x, (sub_t) {neighbor_Y, neighbor_X});
            
            if(in_patch.mat_data[patch_idx] != 0 && isnan(P_xi[in_patch.mat_data[patch_idx]-1])) {
                double neighbor_X_coord = neighbor_X*r_cartesian_mesh_obj->h+r_cartesian_mesh_obj->x_start;
                double neighbor_Y_coord = neighbor_Y*r_cartesian_mesh_obj->h+r_cartesian_mesh_obj->y_start;
                rd_mat_t initial_guess_xi = rd_mat_init(patch_XI_data+i, 1, 1);
                rd_mat_t initial_guess_eta = rd_mat_init(patch_ETA_data+i, 1, 1);
                inverse_M_p_return_type_t xi_eta = q_patch_inverse_M_p(q_patch, neighbor_X_coord, neighbor_Y_coord, &initial_guess_xi, &initial_guess_eta);

                if (xi_eta.converged) {
                    P_xi[in_patch.mat_data[patch_idx]-1] = xi_eta.xi;
                    P_eta[in_patch.mat_data[patch_idx]-1] = xi_eta.eta;
                }
                else {
                    printf("Nonconvergence in interpolation!!!");
                }
            }
        }
    }

    MKL_INT nan_count;
    bool first_iter = true;
    while (true) {
        if (!first_iter && nan_count == 0) {
            break;
        }
        nan_count = 0;
        for (int i = 0; i < n_in_patch; i++) {
            if (isnan(P_xi[i])) {
                bool is_touched = false;
                sub_t idx = ind2sub(r_cartesian_mesh_obj->n_y, r_cartesian_mesh_obj->n_x, r_patch_idxs_data[i]);

                MKL_INT neighbor_shift_x[8] = {1,-1, 0, 0, 1, 1, -1, -1};
                MKL_INT neighbor_shift_y[8] = {0, 0, -1, 1, 1, -1, 1, -1};
                for (int j = 0; j < 8; j++) {
                    MKL_INT neighbor_i = idx.i + neighbor_shift_y[j];
                    MKL_INT neighbor_j = idx.j + neighbor_shift_x[j];
                    
                    if (neighbor_j > r_cartesian_mesh_obj->n_x-1 || neighbor_j < 0 || neighbor_i > r_cartesian_mesh_obj->n_y-1 || neighbor_i < 0) {
                        continue;
                    }

                    MKL_INT neighbor = sub2ind(r_cartesian_mesh_obj->n_y, r_cartesian_mesh_obj->n_x, (sub_t) {neighbor_i, neighbor_j});
                    if (in_patch.mat_data[neighbor] != 0 && !isnan(P_xi[in_patch.mat_data[neighbor]-1])) {
                        double idx_x_coord = idx.j*r_cartesian_mesh_obj->h + r_cartesian_mesh_obj->x_start;
                        double idx_y_coord = idx.i*r_cartesian_mesh_obj->h + r_cartesian_mesh_obj->y_start;
                        rd_mat_t initial_guess_xi = rd_mat_init(P_xi + in_patch.mat_data[neighbor] - 1, 1, 1);
                        rd_mat_t initial_guess_eta = rd_mat_init(P_eta + in_patch.mat_data[neighbor] - 1, 1, 1);
                        inverse_M_p_return_type_t xi_eta = q_patch_inverse_M_p(q_patch, idx_x_coord, idx_y_coord, &initial_guess_xi, &initial_guess_eta);

                        if (xi_eta.converged) {
                            P_xi[i] = xi_eta.xi;
                            P_eta[i] = xi_eta.eta;
                            is_touched = true;
                        }
                        else {
                            printf("Nonconvergence in interpolation!!!");
                        }
                    }
                }
                if (!is_touched) {
                    nan_count += 1;
                }
            }
        }
        first_iter = false;
    }

    double *f_R_patch = malloc(n_in_patch * sizeof(double));
    for (int i = 0; i < n_in_patch; i++) {
        double xi_point = P_xi[in_patch.mat_data[r_patch_idxs.mat_data[i]]-1];
        double eta_point = P_eta[in_patch.mat_data[r_patch_idxs.mat_data[i]]-1];

        locally_compute_return_type_t f_locally_compute = q_patch_locally_compute(q_patch, xi_point, eta_point, M);
        if(f_locally_compute.in_range) {
            f_R_patch[i] = f_locally_compute.f_xy;	      
        }
	else {
            printf("WARNING: interpolating point not in patch\n");
	}
    }
    
    for (int i = 0; i < n_in_patch; i++) {
        r_cartesian_mesh_obj->f_R->mat_data[r_patch_idxs.mat_data[i]] += f_R_patch[i];
    }

    free(bound_X_data);
    free(bound_Y_data);
    free(in_patch_data);
    free(r_patch_idxs_data);
    free(patch_XI_data);
    free(patch_ETA_data);
    free(patch_X_data);
    free(patch_Y_data);
    free(floor_X_j);
    free(ceil_X_j);
    free(floor_Y_j);
    free(ceil_Y_j);
    free(P_xi);
    free(P_eta);
    free(f_R_patch);
}

void r_cartesian_mesh_fill_interior(r_cartesian_mesh_obj_t *r_cartesian_mesh_obj, scalar_func_2D_t f) {
    for (int i = 0; i < r_cartesian_mesh_obj->n_x * r_cartesian_mesh_obj->n_y; i++) {
        if (r_cartesian_mesh_obj->in_interior->mat_data[i]) {
            r_cartesian_mesh_obj->f_R->mat_data[i] = f(r_cartesian_mesh_obj->R_X->mat_data[i], r_cartesian_mesh_obj->R_Y->mat_data[i]);
        }
    }
}

static inline MKL_Complex16 z(double r, double i) {MKL_Complex16 z; z.real = r; z.imag = i; return z;}

double r_cartesian_mesh_compute_fc_error(r_cartesian_mesh_obj_t *r_cartesian_mesh_obj, scalar_func_2D_t f, MKL_INT rho_err, rd_mat_t boundary_X, rd_mat_t boundary_Y, MKL_INT n_x_fft, MKL_INT n_y_fft) {
    // for readability
    double n_x_h = ((double) n_x_fft)/2;
    double n_y_h = ((double) n_y_fft)/2;

    // Timing the FFT buffer fill + FFT
    clock_t start_fft, end_fft;
    start_fft = clock();

    // converts real numbers to complex and also row-major ordering
    MKL_Complex16 buf[n_y_fft][n_x_fft];
    memset(buf, 0, sizeof(buf));

    for (int i = 0; i < r_cartesian_mesh_obj->n_y; i++) {
        for (int j = 0; j < r_cartesian_mesh_obj->n_x; j++) {
            buf[i][j] = z(r_cartesian_mesh_obj->f_R->mat_data[sub2ind(r_cartesian_mesh_obj->n_y, r_cartesian_mesh_obj->n_x, (sub_t) {i, j})], 0.0);
        }
    }

    DFTI_DESCRIPTOR_HANDLE f_hand;
    MKL_LONG f_dims[2] = {n_y_fft, n_x_fft};
    DftiCreateDescriptor(&f_hand, DFTI_DOUBLE, DFTI_COMPLEX, 2, f_dims);
    DftiSetValue(f_hand, DFTI_FORWARD_SCALE, 1.0/(n_x_fft*n_y_fft));
    DftiCommitDescriptor(f_hand);
    DftiComputeForward(f_hand, buf);

    end_fft = clock();
    printf("[FC error] FFT timing: %f seconds\n", ((double)(end_fft - start_fft)) / CLOCKS_PER_SEC);

    DftiFreeDescriptor(&f_hand);

    // computes error meshes
    double h_err = r_cartesian_mesh_obj->h / (double) rho_err;
    double x_err_end = r_cartesian_mesh_obj->x_start + n_x_fft * r_cartesian_mesh_obj->h - h_err;
    double y_err_end = r_cartesian_mesh_obj->y_start + n_y_fft * r_cartesian_mesh_obj->h - h_err;
    MKL_INT n_x_err = round((x_err_end -r_cartesian_mesh_obj->x_start)/h_err) + 1;
    MKL_INT n_y_err = round((y_err_end -r_cartesian_mesh_obj->y_start)/h_err) + 1;

    double x_err_mesh_data[n_x_err];
    double y_err_mesh_data[n_y_err];
    rd_mat_t x_err_mesh = rd_mat_init(x_err_mesh_data, n_x_err, 1);
    rd_mat_t y_err_mesh = rd_mat_init(y_err_mesh_data, n_y_err, 1);
    rd_linspace(r_cartesian_mesh_obj->x_start, x_err_end, n_x_err, &x_err_mesh);
    rd_linspace(r_cartesian_mesh_obj->y_start, y_err_end, n_y_err, &y_err_mesh);

    // computes 2d error mesh
    double R_X_err_data[n_y_err*n_x_err];
    double R_Y_err_data[n_y_err*n_x_err];
    rd_mat_t R_X_err = rd_mat_init_no_shape(R_X_err_data);
    rd_mat_t R_Y_err = rd_mat_init_no_shape(R_Y_err_data);
    rd_meshgrid(x_err_mesh, y_err_mesh, &R_X_err, &R_Y_err);

    // computes interior mask for this error mesh
    MKL_INT in_interior_err_data[n_y_err*n_x_err];
    ri_mat_t in_interior_err = ri_mat_init_no_shape(in_interior_err_data);
    inpolygon_mesh(R_X_err, R_Y_err, boundary_X, boundary_Y, &in_interior_err);

    MKL_INT n_x_diff = n_x_err - n_x_fft;
    MKL_INT n_y_diff = n_y_err - n_y_fft;

    // creating arrays of zeroes that will have relevant fft coefffs filled in
    MKL_Complex16 buf_padded[n_y_err][n_x_err];
    memset(buf_padded, 0, sizeof(buf_padded));

    // padding q1
    for(int i = 0; i < (int) ceil(n_y_h); i++) {
        for (int j = 0; j < (int) ceil(n_x_h); j++) {
            buf_padded[i][j] = buf[i][j];
        }
    }

    // padding q2
    for(int i = 0; i < ceil(n_y_h); i++) {
        for (int j = ceil(n_x_h); j < n_x_fft; j++) {
            buf_padded[i][n_x_diff+j] = buf[i][j];
        }
    }

    // padding q3
    for(int i = ceil(n_y_h); i < n_y_fft; i++) {
        for (int j = 0; j < ceil(n_x_h); j++) {
            buf_padded[n_y_diff+i][j] = buf[i][j];
        }
    }

    // padding q4
    for(int i = ceil(n_y_h); i < n_y_fft; i++) {
        for (int j = ceil(n_x_h); j < n_x_fft; j++) {
            buf_padded[n_y_diff+i][n_x_diff+j] = buf[i][j];
        }
    }

    // performs backward fft
    DFTI_DESCRIPTOR_HANDLE b_hand;
    MKL_LONG b_dims[2] = {n_y_err, n_x_err};
    DftiCreateDescriptor(&b_hand, DFTI_DOUBLE, DFTI_COMPLEX, 2, b_dims);
    DftiCommitDescriptor(b_hand);
    DftiComputeBackward(b_hand, buf_padded);
    DftiFreeDescriptor(&b_hand);

    double intp_fc[n_y_err*n_x_err];
    for (int i = 0; i < n_y_err; i++) {
        for (int j = 0; j < n_x_err; j++) {
        intp_fc[sub2ind(n_y_err, n_x_err, (sub_t){i, j})] = buf_padded[i][j].real;
        }
    }

    double f_max = 0;
    for (int i = 0; i < n_y_err*n_x_err; i++) {
        if (in_interior_err_data[i]) {
	  f_max = fmax(f_max, f(R_X_err_data[i], R_Y_err_data[i]));
        }
    }
    
    double fc_err = 0;
    for (int i = 0; i < n_y_err*n_x_err; i++) {
        if (in_interior_err_data[i]) {
            fc_err = fmax(fc_err, fabs(f(R_X_err_data[i], R_Y_err_data[i]) - intp_fc[i]));
        }
    }

    printf("Absolute maximum error: %e\n", fc_err);
    printf("Relative maximum error: %e\n", fc_err/f_max);

    double f_l2 = 0;
    for (int i = 0; i < n_y_err*n_x_err; i++) {
        if (in_interior_err_data[i]) {
	  f_l2 += pow(f(R_X_err_data[i], R_Y_err_data[i]), 2);
        }
    }

    double fc_err_2 = 0;
    for (int i = 0; i < n_y_err*n_x_err; i++) {
        if (in_interior_err_data[i]) {
	  fc_err_2 += pow(f(R_X_err_data[i], R_Y_err_data[i]) - intp_fc[i], 2);
        }
    }    

    printf("Relative l2 error: %e\n", sqrt(fc_err_2)/sqrt(f_l2));
    
    return fc_err;
}

double r_cartesian_mesh_compute_fc_error_heap(r_cartesian_mesh_obj_t *r_cartesian_mesh_obj, scalar_func_2D_t f, MKL_INT rho_err, rd_mat_t boundary_X, rd_mat_t boundary_Y, MKL_INT n_x_fft, MKL_INT n_y_fft) {
    // for readability
    double n_x_h = ((double) n_x_fft)/2;
    double n_y_h = ((double) n_y_fft)/2;

    // Timing the FFT buffer fill + FFT
    clock_t start_fft, end_fft;
    start_fft = clock();

    // converts real numbers to complex and also row-major ordering
    // Note: buf is stored in row-major order (i*n_x_fft + j), matching original 2D array behavior
    MKL_Complex16 *buf = malloc(n_y_fft*n_x_fft * sizeof(MKL_Complex16));
    if (!buf) {
        fprintf(stderr, "Error: malloc failed for buf\n");
        return 0.0;
    }
    memset(buf, 0, n_y_fft*n_x_fft * sizeof(MKL_Complex16));

    for (int i = 0; i < r_cartesian_mesh_obj->n_y; i++) {
        for (int j = 0; j < r_cartesian_mesh_obj->n_x; j++) {
            buf[i*n_x_fft + j] = z(r_cartesian_mesh_obj->f_R->mat_data[sub2ind(r_cartesian_mesh_obj->n_y, r_cartesian_mesh_obj->n_x, (sub_t) {i, j})], 0.0);
        }
    }

    DFTI_DESCRIPTOR_HANDLE f_hand;
    MKL_LONG f_dims[2] = {n_y_fft, n_x_fft};
    DftiCreateDescriptor(&f_hand, DFTI_DOUBLE, DFTI_COMPLEX, 2, f_dims);
    DftiSetValue(f_hand, DFTI_FORWARD_SCALE, 1.0/(n_x_fft*n_y_fft));
    DftiCommitDescriptor(f_hand);
    DftiComputeForward(f_hand, buf);

    end_fft = clock();
    printf("[FC error] FFT timing: %f seconds\n", ((double)(end_fft - start_fft)) / CLOCKS_PER_SEC);

    DftiFreeDescriptor(&f_hand);

    // computes error meshes
    double h_err = r_cartesian_mesh_obj->h / (double) rho_err;
    double x_err_end = r_cartesian_mesh_obj->x_start + n_x_fft * r_cartesian_mesh_obj->h - h_err;
    double y_err_end = r_cartesian_mesh_obj->y_start + n_y_fft * r_cartesian_mesh_obj->h - h_err;
    MKL_INT n_x_err = round((x_err_end -r_cartesian_mesh_obj->x_start)/h_err) + 1;
    MKL_INT n_y_err = round((y_err_end -r_cartesian_mesh_obj->y_start)/h_err) + 1;

    double *x_err_mesh_data = malloc(n_x_err * sizeof(double));
    double *y_err_mesh_data = malloc(n_y_err * sizeof(double));
    if (!x_err_mesh_data || !y_err_mesh_data) {
        fprintf(stderr, "Error: malloc failed for x_err_mesh_data or y_err_mesh_data\n");
        free(buf);
        if (x_err_mesh_data) free(x_err_mesh_data);
        if (y_err_mesh_data) free(y_err_mesh_data);
        return 0.0;
    }
    rd_mat_t x_err_mesh = rd_mat_init(x_err_mesh_data, n_x_err, 1);
    rd_mat_t y_err_mesh = rd_mat_init(y_err_mesh_data, n_y_err, 1);
    rd_linspace(r_cartesian_mesh_obj->x_start, x_err_end, n_x_err, &x_err_mesh);
    rd_linspace(r_cartesian_mesh_obj->y_start, y_err_end, n_y_err, &y_err_mesh);

    // computes 2d error mesh
    double *R_X_err_data = malloc(n_y_err*n_x_err * sizeof(double));
    double *R_Y_err_data = malloc(n_y_err*n_x_err * sizeof(double));
    if (!R_X_err_data || !R_Y_err_data) {
        fprintf(stderr, "Error: malloc failed for R_X_err_data or R_Y_err_data\n");
        free(buf);
        free(x_err_mesh_data);
        free(y_err_mesh_data);
        if (R_X_err_data) free(R_X_err_data);
        if (R_Y_err_data) free(R_Y_err_data);
        return 0.0;
    }
    rd_mat_t R_X_err = rd_mat_init_no_shape(R_X_err_data);
    rd_mat_t R_Y_err = rd_mat_init_no_shape(R_Y_err_data);
    rd_meshgrid(x_err_mesh, y_err_mesh, &R_X_err, &R_Y_err);

    // computes interior mask for this error mesh
    MKL_INT *in_interior_err_data = malloc(n_y_err*n_x_err * sizeof(MKL_INT));
    if (!in_interior_err_data) {
        fprintf(stderr, "Error: malloc failed for in_interior_err_data\n");
        free(buf);
        free(x_err_mesh_data);
        free(y_err_mesh_data);
        free(R_X_err_data);
        free(R_Y_err_data);
        return 0.0;
    }
    ri_mat_t in_interior_err = ri_mat_init_no_shape(in_interior_err_data);
    inpolygon_mesh(R_X_err, R_Y_err, boundary_X, boundary_Y, &in_interior_err);

    MKL_INT n_x_diff = n_x_err - n_x_fft;
    MKL_INT n_y_diff = n_y_err - n_y_fft;

    // creating arrays of zeroes that will have relevant fft coefffs filled in
    // buf_padded is stored in row-major order (i*n_x_err + j)
    MKL_Complex16 *buf_padded = malloc(n_y_err*n_x_err * sizeof(MKL_Complex16));
    if (!buf_padded) {
        fprintf(stderr, "Error: malloc failed for buf_padded\n");
        free(x_err_mesh_data);
        free(y_err_mesh_data);
        free(R_X_err_data);
        free(R_Y_err_data);
        free(in_interior_err_data);
        free(buf);
        return 0.0;
    }
    memset(buf_padded, 0, n_y_err*n_x_err * sizeof(MKL_Complex16));

    // padding q1
    for(int i = 0; i < (int) ceil(n_y_h); i++) {
        for (int j = 0; j < (int) ceil(n_x_h); j++) {
            buf_padded[i*n_x_err + j] = buf[i*n_x_fft + j];
        }
    }

    // padding q2
    for(int i = 0; i < ceil(n_y_h); i++) {
        for (int j = ceil(n_x_h); j < n_x_fft; j++) {
            buf_padded[i*n_x_err + (n_x_diff+j)] = buf[i*n_x_fft + j];
        }
    }

    // padding q3
    for(int i = ceil(n_y_h); i < n_y_fft; i++) {
        for (int j = 0; j < ceil(n_x_h); j++) {
            buf_padded[(n_y_diff+i)*n_x_err + j] = buf[i*n_x_fft + j];
        }
    }

    // padding q4
    for(int i = ceil(n_y_h); i < n_y_fft; i++) {
        for (int j = ceil(n_x_h); j < n_x_fft; j++) {
            buf_padded[(n_y_diff+i)*n_x_err + (n_x_diff+j)] = buf[i*n_x_fft + j];
        }
    }

    // performs backward fft
    DFTI_DESCRIPTOR_HANDLE b_hand;
    MKL_LONG b_dims[2] = {n_y_err, n_x_err};
    DftiCreateDescriptor(&b_hand, DFTI_DOUBLE, DFTI_COMPLEX, 2, b_dims);
    DftiCommitDescriptor(b_hand);
    DftiComputeBackward(b_hand, buf_padded);
    DftiFreeDescriptor(&b_hand);

    double *intp_fc = malloc(n_y_err*n_x_err * sizeof(double));
    if (!intp_fc) {
        fprintf(stderr, "Error: malloc failed for intp_fc\n");
        free(x_err_mesh_data);
        free(y_err_mesh_data);
        free(R_X_err_data);
        free(R_Y_err_data);
        free(in_interior_err_data);
        free(buf);
        free(buf_padded);
        return 0.0;
    }
    for (int i = 0; i < n_y_err; i++) {
        for (int j = 0; j < n_x_err; j++) {
            // sub2ind returns column-major index, buf_padded is row-major, so use row-major indexing
            intp_fc[sub2ind(n_y_err, n_x_err, (sub_t){i, j})] = buf_padded[i*n_x_err + j].real;
        }
    }

    double f_max = 0;
    for (int i = 0; i < n_y_err*n_x_err; i++) {
        if (in_interior_err_data[i]) {
	  f_max = fmax(f_max, f(R_X_err_data[i], R_Y_err_data[i]));
        }
    }
    
    double fc_err = 0;
    for (int i = 0; i < n_y_err*n_x_err; i++) {
        if (in_interior_err_data[i]) {
            fc_err = fmax(fc_err, fabs(f(R_X_err_data[i], R_Y_err_data[i]) - intp_fc[i]));
        }
    }

    printf("Absolute maximum error: %e\n", fc_err);
    printf("Relative maximum error: %e\n", fc_err/f_max);

    double f_l2 = 0;
    for (int i = 0; i < n_y_err*n_x_err; i++) {
        if (in_interior_err_data[i]) {
	  f_l2 += pow(f(R_X_err_data[i], R_Y_err_data[i]), 2);
        }
    }

    double fc_err_2 = 0;
    for (int i = 0; i < n_y_err*n_x_err; i++) {
        if (in_interior_err_data[i]) {
	  fc_err_2 += pow(f(R_X_err_data[i], R_Y_err_data[i]) - intp_fc[i], 2);
        }
    }    

    printf("Relative l2 error: %e\n", sqrt(fc_err_2)/sqrt(f_l2));
    
    free(x_err_mesh_data);
    free(y_err_mesh_data);
    free(R_X_err_data);
    free(R_Y_err_data);
    free(in_interior_err_data);
    free(buf);
    free(buf_padded);
    free(intp_fc);
    return fc_err;
}

MKL_INT inpolygon_mesh(rd_mat_t R_X, rd_mat_t R_Y, rd_mat_t boundary_X, rd_mat_t boundary_Y, ri_mat_t *in_msk) {
    ri_mat_shape(in_msk, R_X.rows, R_X.columns);
    memset(in_msk->mat_data, 0, in_msk->rows*in_msk->columns*sizeof(MKL_INT));

    MKL_INT n_edges = boundary_X.rows-1;

    rd_mat_t boundary_x_edge_1 = rd_mat_init(boundary_X.mat_data, n_edges, 1);
    rd_mat_t boundary_x_edge_2 = rd_mat_init(boundary_X.mat_data+1, n_edges, 1);
    rd_mat_t boundary_y_edge_1 = rd_mat_init(boundary_Y.mat_data, n_edges, 1);
    rd_mat_t boundary_y_edge_2 = rd_mat_init(boundary_Y.mat_data+1, n_edges, 1);

    MKL_INT boundary_idx_data[n_edges];
    ri_mat_t boundary_idx = ri_mat_init(boundary_idx_data, n_edges, 1);
    ri_range(0, 1, n_edges-1, &boundary_idx);

    double x_start = R_X.mat_data[0];
    double y_start = R_Y.mat_data[0];
    double h_x = R_X.mat_data[R_X.rows] - R_X.mat_data[0];
    double h_y = R_Y.mat_data[1] - R_Y.mat_data[0];

    double boundary_y_j_data[boundary_Y.rows];
    vdSubI(boundary_Y.rows, boundary_Y.mat_data, 1, &y_start, 0, boundary_y_j_data, 1);
    vdDivI(boundary_Y.rows, boundary_y_j_data, 1, &h_y, 0, boundary_y_j_data, 1);

    rd_mat_t boundary_y_edge_1_j = rd_mat_init(boundary_y_j_data, n_edges, 1);
    rd_mat_t boundary_y_edge_2_j = rd_mat_init(boundary_y_j_data+1, n_edges, 1);

    MKL_INT intersection_edges_msk[n_edges];
    MKL_INT n_intersection_edges = 0;
    for (int i = 0; i < n_edges; i++ ) {
        if(fabs(boundary_y_edge_1_j.mat_data[i] - round(boundary_y_edge_1_j.mat_data[i])) < __DBL_EPSILON__) {
            boundary_y_edge_1_j.mat_data[i] = round(boundary_y_edge_1_j.mat_data[i]);
        }
        if(fabs(boundary_y_edge_2_j.mat_data[i] - round(boundary_y_edge_2_j.mat_data[i])) < __DBL_EPSILON__) {
            boundary_y_edge_2_j.mat_data[i] = round(boundary_y_edge_2_j.mat_data[i]);
        }

        intersection_edges_msk[i] = floor(boundary_y_edge_1_j.mat_data[i]) != floor(boundary_y_edge_2_j.mat_data[i]);
        if (((MKL_INT) floor(boundary_y_edge_1_j.mat_data[i])) != ((MKL_INT) floor(boundary_y_edge_2_j.mat_data[i]))) {
            n_intersection_edges += 1;
        }
    }
    MKL_INT intersection_idxs_data[n_intersection_edges];
    MKL_INT curr_idx = 0;
    for (int i = 0; i < n_edges; i++) {
        if(intersection_edges_msk[i]) {
            intersection_idxs_data[curr_idx] = boundary_idx_data[i];
            curr_idx += 1;
        }
    }
    
    for (int i = 0; i < n_intersection_edges; i++) {
        MKL_INT intersection_idx = intersection_idxs_data[i];

        double x_edge_1 = boundary_x_edge_1.mat_data[intersection_idx];
        double x_edge_2 = boundary_x_edge_2.mat_data[intersection_idx];
        double y_edge_1 = boundary_y_edge_1.mat_data[intersection_idx];
        double y_edge_2 = boundary_y_edge_2.mat_data[intersection_idx];
        MKL_INT y_edge_1_j = (MKL_INT) floor(boundary_y_edge_1_j.mat_data[intersection_idx]);
        MKL_INT y_edge_2_j = (MKL_INT) floor(boundary_y_edge_2_j.mat_data[intersection_idx]);

        MKL_INT intersection_mesh_length = MAX(y_edge_1_j, y_edge_2_j) - MIN(y_edge_1_j, y_edge_2_j);
        MKL_INT intersection_mesh_y_j_data[intersection_mesh_length];
        ri_mat_t intersection_mesh_y_j = ri_mat_init(intersection_mesh_y_j_data, intersection_mesh_length, 1);
        ri_range(MIN(y_edge_1_j, y_edge_2_j)+1, 1, MAX(y_edge_1_j, y_edge_2_j), &intersection_mesh_y_j);

        for (int j = 0; j < intersection_mesh_length; j++) {
            double intersection_y = intersection_mesh_y_j_data[j] * h_y + y_start;
            double intersection_x = x_edge_1 + (x_edge_2-x_edge_1)*(intersection_y-y_edge_1)/(y_edge_2-y_edge_1);

            MKL_INT mesh_intersection_idx = sub2ind(in_msk->rows, in_msk->columns, (sub_t) {(MKL_INT) round((intersection_y-y_start)/h_y), (MKL_INT) floor((intersection_x-x_start)/h_x)});
            in_msk->mat_data[mesh_intersection_idx] = !in_msk->mat_data[mesh_intersection_idx];
        }
    }


    MKL_INT n_points_interior = 0;
    for(int row_idx = 0; row_idx < in_msk->rows; row_idx++) {
        bool in_interior = false;
        for (int col_idx = 0; col_idx < in_msk->columns; col_idx++) {
            MKL_INT idx = sub2ind(in_msk->rows, in_msk->columns, (sub_t) {row_idx, col_idx});
            if (in_msk->mat_data[idx] && !in_interior) {
                in_interior = true;
                in_msk->mat_data[idx] = 0;
            } else if (in_msk->mat_data[idx] && in_interior){
                in_interior = false;
                n_points_interior += 1;
            }
            else if (in_interior) {
                in_msk->mat_data[idx] = 1;
                n_points_interior += 1;
            }
        }
    }

    return n_points_interior;
}
