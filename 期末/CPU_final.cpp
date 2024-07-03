#include<iostream>
#include <stdio.h>
#include<cstring>
#include<typeinfo>
#include <stdlib.h>
#include<cmath>
#include<mpi.h>
#include<windows.h>
#include<omp.h>
#include<tmmintrin.h>
#include<xmmintrin.h>
#include<emmintrin.h>
#include<pmmintrin.h>
#include<smmintrin.h>
#include<nmmintrin.h>
#include<immintrin.h>

using namespace std;
#define N 1500
#define NUM_THREADS 7
float** A = NULL;

long long head, tail, freq;

void A_init() {  
    A = new float* [N];
    for (int i = 0; i < N; i++) {
        A[i] = new float[N];
    }
    for (int i = 0; i < N; i++) {
        A[i][i] = 1.0;
        for (int j = i + 1; j < N; j++) {
            A[i][j] = rand() % 5000;
        }

    }
    for (int k = 0; k < N; k++) {
        for (int i = k + 1; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i][j] += A[k][j];
                A[i][j] = (int)A[i][j] % 5000;
            }
        }
    }
}
void A_initAsEmpty() {
    A = new float* [N];
    for (int i = 0; i < N; i++) {
        A[i] = new float[N];
        memset(A[i], 0, N * sizeof(float));
    }

}

void deleteA() {
    for (int i = 0; i < N; i++) {
        delete[] A[i];
    }
    delete A;
}


void LU() {    
    for (int k = 0; k < N; k++) {
        for (int j = k + 1; j < N; j++) {
            A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1.0;

        for (int i = k + 1; i < N; i++) {
            for (int j = k + 1; j < N; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}



double CPU_final(int argc, char* argv[]) { 
    double start_time = 0;
    double end_time = 0;
    MPI_Init(&argc, &argv);
    cout << MPI_Wtick();
    int total = 0;
    int rank = 0;
    int i = 0;
    int j = 0;
    int k = 0;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &total);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int begin = N / total * rank;
    int end = (rank == total - 1) ? N : N / total * (rank + 1);

    if (rank == 0) { 
        A_init();
        MPI_Request* request = new MPI_Request[N - end];
        for (j = 1; j < total; j++) {
            int b = j * (N / total), e = (j == total - 1) ? N : (j + 1) * (N / total);

            for (i = b; i < e; i++) {
                MPI_Isend(&A[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD, &request[i - end]);
            }

        }
        MPI_Waitall(N - end, request, MPI_STATUS_IGNORE); 

    }
    else {
        A_initAsEmpty();
        MPI_Request* request = new MPI_Request[end - begin];
        for (i = begin; i < end; i++) {
            MPI_Irecv(&A[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &request[i - begin]);  
        }
        MPI_Waitall(end - begin, request, MPI_STATUS_IGNORE);

    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
#pragma omp parallel  num_threads(NUM_THREADS),private(i,j,k)
    for (k = 0; k < N; k++) {
#pragma omp single
        {
            if ((begin <= k && k < end)) {
                j = k + 1;
                while ((int)A[k][j] % 32) {
                    A[k][j] = A[k][j] / A[k][k];  
                    j++;
                }
                __m256 t1 = _mm256_set1_ps(A[k][k]);
                for (; j + 8 <= N; j += 8) {
                    __m256 t2 = _mm256_load_ps(&A[k][j]);
                    t2 = _mm256_div_ps(t2, t1);
                    _mm256_store_ps(&A[k][j], t2);
                }
                for (; j < N; j++) {
                    A[k][j] = A[k][j] / A[k][k];
                }
                A[k][k] = 1.0;
                MPI_Request* request = new MPI_Request[total - 1 - rank];
                for (j = rank + 1; j < total; j++) { 

                    MPI_Isend(&A[k][0], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD, &request[j - rank - 1]);
                }
                MPI_Waitall(total - 1 - rank, request, MPI_STATUS_IGNORE);
                if (k == end - 1)
                    break;
            }
            else {
                int src = k / (N / total);
                MPI_Request request;
                MPI_Irecv(&A[k][0], N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &request);
                MPI_Wait(&request, MPI_STATUS_IGNORE); 
            }
        }
#pragma omp for schedule(guided) 
        for (i = max(begin, k + 1); i < end; i++) {
            j = k + 1;
            while ((int)A[i][j] % 32)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
                j++;
            }

            __m256 vik = _mm256_set1_ps(A[i][k]);
            for (; j + 8 <= N; j += 8) {
                __m256 vkj = _mm256_loadu_ps(&A[k][j]); 
                __m256 vij = _mm256_load_ps(&A[i][j]);
                __m256 vx = _mm256_mul_ps(vik, vkj);
                vij = _mm256_sub_ps(vij, vx);
                _mm256_store_ps(&A[i][j], vij); 
            }
            for (; j < N; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j]; 
            }
            A[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == total - 1) {
        end_time = MPI_Wtime();
        printf("CPU最终优化耗时：%.4lf ms\n", 1000 * (end_time - start_time));
    }
    MPI_Finalize();
    return end_time - start_time;

}

void cal(void(*func)()) {
    A_init();
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    func();
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);

}

int main(int argc, char* argv[]) {
    /*   QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
       cal(LU);
       cout << "平凡算法耗时：" << (tail - head) * 1000 / freq << "ms" << endl;
       deleteA();*/

    CPU_final(argc, argv);

}