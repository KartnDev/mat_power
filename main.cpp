#pragma clang diagnostic push
#pragma ide diagnostic ignored "openmp-use-default-none"

#include <omp.h>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <iostream>
#include <fstream>

constexpr int NUM_THREADS = 24;


void parallel_matmul(const double *a, int row_a, int col_a, const double *b, int row_b, int col_b, double *result) {
    if (col_a != row_b) {
        return;
    }
    int i, j, k;
    int index;
    int border = row_a * col_b;
    i = 0;
    j = 0;

#pragma omp parallel for private(i, j, k) num_threads(NUM_THREADS)
    for (index = 0; index < border; index++) {
        i = index / col_b;
        j = index % col_b;
        int row_i = i * col_a;
        int row_c = i * col_b;
        result[row_c + j] = 0;
        for (k = 0; k < row_b; k++) {
            result[row_c + j] += a[row_i + k] * b[k * col_b + j];
        }
    }
}

void parallel_matrix_power(const double *matrix, int matrix_size, double *result, int power) {
    double *temp = (double *) malloc(sizeof(double) * matrix_size * matrix_size);

    for (int i = 0; i < matrix_size; ++i) {
        for (int j = 0; j < matrix_size; ++j) {
            temp[i * matrix_size + j] = matrix[i * matrix_size + j];
        }
    }

    for (int iter = 1; iter < power; ++iter) {
        parallel_matmul(matrix, matrix_size, matrix_size, temp, matrix_size, matrix_size, result);

        for (int i = 0; i < matrix_size; ++i) {
            for (int j = 0; j < matrix_size; ++j) {
                temp[i * matrix_size + j] = result[i * matrix_size + j];
            }
        }
    }
}

void one_thread_matmul(const double *a, int row_a, int col_a, const double *b, int row_b, int col_b, double *result) {
    if (col_a != row_b) {
        return;
    }
    int i, j, k;
    int index;
    int border = row_a * col_b;
    i = 0;
    j = 0;

    for (index = 0; index < border; index++) {
        i = index / col_b;
        j = index % col_b;
        int row_i = i * col_a;
        int row_c = i * col_b;
        result[row_c + j] = 0;
        for (k = 0; k < row_b; k++) {
            result[row_c + j] += a[row_i + k] * b[k * col_b + j];
        }
    }
}


void one_thread_power(const double *matrix, int matrix_size, double *result, int power) {
    double *temp = (double *) (malloc(sizeof(double) * matrix_size * matrix_size));

    for (int i = 0; i < matrix_size; ++i) {
        for (int j = 0; j < matrix_size; ++j) {
            temp[i * matrix_size + j] = matrix[i * matrix_size + j];
        }
    }

    for (int iter = 1; iter < power; ++iter) {
        one_thread_matmul(matrix, matrix_size, matrix_size, temp, matrix_size, matrix_size, result);

        for (int i = 0; i < matrix_size; ++i) {
            for (int j = 0; j < matrix_size; ++j) {
                temp[i * matrix_size + j] = result[i * matrix_size + j];
            }
        }
    }
}

void write_matrix(std::string filename, double *mat, int size) {
    std::ofstream fileDescriptor;
    fileDescriptor.open(filename);

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            fileDescriptor << mat[i * size + j] << " ";
        }
        fileDescriptor << "\n";
    }

    fileDescriptor.close();
}


int main() {
    int matrix_size = 3;
    int power = 3;

    // random seed
    srand(time(nullptr));

    double *matrix, *parallel_result, *one_thread_result;

    matrix = (double *) (malloc(sizeof(double) * matrix_size * matrix_size));
    parallel_result = (double *) (malloc(sizeof(double) * matrix_size * matrix_size));
    one_thread_result = (double *) (malloc(sizeof(double) * matrix_size * matrix_size));

    // random initialize matrix A
    for (int i = 0; i < matrix_size; ++i) {
        for (int j = 0; j < matrix_size; ++j) {
            matrix[i * matrix_size + j] = (((double) rand() / (RAND_MAX)) + 1) * 10;
        }
    }
    write_matrix("matrix.txt", matrix, matrix_size);

    // start the OMP PARALLEL version
    auto started = std::chrono::high_resolution_clock::now();
    parallel_matrix_power(matrix, matrix_size, parallel_result, power);
    auto done = std::chrono::high_resolution_clock::now();
    auto parallel_elapsed_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(done - started).count();

    // start the ONE THREAD version
    started = std::chrono::high_resolution_clock::now();
    one_thread_power(matrix, matrix_size, one_thread_result, power);
    done = std::chrono::high_resolution_clock::now();
    auto seq_elapsed_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(done - started).count();

    std::cout << "Sequential speed: " << (double) seq_elapsed_time_ms / 1000 << "s\n";
    std::cout << "Parallel speed: " << (double) parallel_elapsed_time_ms / 1000 << "s\n";
    std::cout << "Speedup: " << (double) seq_elapsed_time_ms / (double) parallel_elapsed_time_ms << "\n";
    std::cout << "Used threads: " << NUM_THREADS << "\n";

    // check for valid data from parallel and one thread result
    bool all_ok = true;
    for (int i = 0; i < matrix_size; ++i) {
        for (int j = 0; j < matrix_size; ++j) {
            // equals between doubles
            if (std::abs(one_thread_result[i * matrix_size + j] - parallel_result[i * matrix_size + j]) > 0.001) {
                all_ok = false;
            }
        }
    }

    // roughly compute speedup
    if (all_ok) {
        write_matrix("output_matrix.txt", parallel_result, matrix_size);
        printf("all results are correct!!!\n");
    } else {
        printf("incorrect results\n");
    }


    return 0;
}

#pragma clang diagnostic pop