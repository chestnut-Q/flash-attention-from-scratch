#include "mpi.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <sys/stat.h> // For mkdir
#include <sys/types.h>
#include <errno.h>

#define MPI_TYPE MPI_FLOAT

extern void square_attention(int lda, float* Q, float* K, float* V, float* Y, int rank, int size);

int test_sizes[] =
    {63, 64, 65, 127, 128, 129, 191, 192, 193, 255, 256, 257, 319, 320, 321, 383, 384, 385, 447, 448, 449, 511, 512, 513, 575, 576, 577, 639, 640, 641, 703, 704, 705, 767, 768, 769, 831, 832, 833, 895, 896, 897, 959, 960, 961, 1023, 1024, 1025, 1087, 1088, 1089, 1151, 1152, 1153, 1215, 1216, 1217, 1279, 1280, 1281, 1343, 1344, 1345, 1407, 1408, 1409, 1471, 1472, 1473, 1535, 1536, 1537, 1599, 1600, 1601, 1663, 1664, 1665, 1727, 1728, 1729, 1791, 1792, 1793, 1855, 1856, 1857, 1919, 1920, 1921, 1983, 1984, 1985, 2047, 2048, 2049, 4095, 4096, 4097, 8191, 8192, 8193};
const int nsizes = sizeof(test_sizes) / sizeof(test_sizes[0]);

#define DEFAULT_MIN 0.0f
#define DEFAULT_MAX 1.0f
#define DEFAULT_OUTPUT_DIR "../data"

void die(const char* message) {
    perror(message);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    exit(EXIT_FAILURE);
}

void generate_random_data(float* arr, int len, float min, float max, int seed) {
    srand(seed);
    for(int i = 0; i < len; ++i) {
        float r = ((float)rand()) / ((float)RAND_MAX);
        arr[i] = min + r * (max - min);
    }
}

void write_to_file(const float* arr, int len, const char* file_path) {
    FILE* file = fopen(file_path, "wb");
    if(file == NULL) {
        perror("Failed to open file for writing");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    size_t written = fwrite(arr, sizeof(float), len, file);
    if(written != (size_t)len) {
        perror("Failed to write all data to file");
        fclose(file);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    fclose(file);
}

int create_directory(const char* path) {
    char tmp[512];
    char* p = NULL;
    size_t len;

    snprintf(tmp, sizeof(tmp), "%s", path);
    len = strlen(tmp);
    if(tmp[len - 1] == '/')
        tmp[len - 1] = 0;
    for(p = tmp + 1; *p; p++) {
        if(*p == '/') {
            *p = 0;
            if(mkdir(tmp, S_IRWXU) != 0) {
                if(errno != EEXIST) {
                    return -1;
                }
            }
            *p = '/';
        }
    }
    if(mkdir(tmp, S_IRWXU) != 0) {
        if(errno != EEXIST) {
            return -1;
        }
    }
    return 0;
}

int main(int argc, char* argv[]) {

    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    float min_val = DEFAULT_MIN;
    float max_val = DEFAULT_MAX;
    const char* output_dir = DEFAULT_OUTPUT_DIR;

    for(int i = 1; i < argc; ++i) {
        if(strcmp(argv[i], "--min") == 0 && i + 1 < argc) {
            min_val = atof(argv[++i]);
        }
        else if(strcmp(argv[i], "--max") == 0 && i + 1 < argc) {
            max_val = atof(argv[++i]);
        }
        else if(strcmp(argv[i], "--output_dir") == 0 && i + 1 < argc) {
            output_dir = argv[++i];
        }
        else if(strcmp(argv[i], "--help") == 0) {
            if (rank == 0) {
                printf("Usage: %s [--min <value>] [--max <value>] [--output_dir <path>]\n", argv[0]);
            }
            MPI_Finalize();
            return 0;
        }
        else {
            if (rank == 0) {
                printf("Unknown argument: %s\n", argv[i]);
                printf("Usage: %s [--min <value>] [--max <value>] [--output_dir <path>]\n", argv[0]);
            }
            MPI_Finalize();
            return EXIT_FAILURE;
        }
    }

    if(rank == 0) {
        printf("Generating test data with min=%.3f, max=%.3f, output_dir=%s\n", min_val, max_val, output_dir);
        
        if(create_directory(output_dir) != 0) {
            die("Failed to create output directory");
        }

        char q_dir[512], k_dir[512], v_dir[512], y_dir[512];
        snprintf(q_dir, sizeof(q_dir), "%s/Q_value", output_dir);
        snprintf(k_dir, sizeof(k_dir), "%s/K_value", output_dir);
        snprintf(v_dir, sizeof(v_dir), "%s/V_value", output_dir);
        snprintf(y_dir, sizeof(y_dir), "%s/output_value", output_dir);

        if(create_directory(q_dir) != 0 ||
           create_directory(k_dir) != 0 ||
           create_directory(v_dir) != 0 ||
           create_directory(y_dir) != 0) {
            die("Failed to create subdirectories");
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    char q_dir[512], k_dir[512], v_dir[512], y_dir[512];
    snprintf(q_dir, sizeof(q_dir), "%s/Q_value", output_dir);
    snprintf(k_dir, sizeof(k_dir), "%s/K_value", output_dir);
    snprintf(v_dir, sizeof(v_dir), "%s/V_value", output_dir);
    snprintf(y_dir, sizeof(y_dir), "%s/output_value", output_dir);

    for(int isize = 0; isize < nsizes; ++isize) {
        int n = test_sizes[isize];
        float* Q = (float*)malloc(sizeof(float) * n * n);
        float* K = (float*)malloc(sizeof(float) * n * n);
        float* V = (float*)malloc(sizeof(float) * n * n);
        float* Y = (float*)malloc(sizeof(float) * n * n);
        if(Q == NULL || K == NULL || V == NULL || Y == NULL) {
            die("Memory allocation failed");
        }

        if(rank == 0) {
            generate_random_data(Q, n*n, min_val, max_val, isize + 1);
            generate_random_data(K, n*n, min_val, max_val, isize + 2);
            generate_random_data(V, n*n, min_val, max_val, isize + 3);
        }

        MPI_Bcast(Q, n*n, MPI_TYPE, 0, MPI_COMM_WORLD);
        MPI_Bcast(K, n*n, MPI_TYPE, 0, MPI_COMM_WORLD);
        MPI_Bcast(V, n*n, MPI_TYPE, 0, MPI_COMM_WORLD);

        square_attention(n, Q, K, V, Y, rank, size);

        if(rank == 0) {
            char file_path[1024];
            snprintf(file_path, sizeof(file_path), "%s/q_%d", q_dir, isize);
            write_to_file(Q, n*n, file_path);
            snprintf(file_path, sizeof(file_path), "%s/k_%d", k_dir, isize);
            write_to_file(K, n*n, file_path);
            snprintf(file_path, sizeof(file_path), "%s/v_%d", v_dir, isize);
            write_to_file(V, n*n, file_path);
            snprintf(file_path, sizeof(file_path), "%s/output_%d", y_dir, isize);
            write_to_file(Y, n*n, file_path);

            printf("Generated data for size %d\n", n);
        }

        free(Q);
        free(K);
        free(V);
        free(Y);
    }

    MPI_Finalize();
    return 0;
}