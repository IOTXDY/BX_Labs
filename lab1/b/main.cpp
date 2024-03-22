#include <iostream>
#include <windows.h>
#include <stdlib.h>

using namespace std;

const int N = 10000; // matrix size

double A[N][N], B[N], res[N];

void init(int n)
{
	for (int i = 0; i < N; i++)
	{
		B[i] = i;
		for (int j = 0; j < N; j++)
			A[i][j] = i + j;
	}
}
int main()
{
	long long head, tail, freq; // timers
	init(N);
	// similar to CLOCKS_PER_SEC
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	// start time
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	for(int t=0;t<20;t++)
    {
        //平凡算法
//        for (int i = 0; i < N; i++)
//        {
//            res[i] = 0.0;
//            for (int j = 0; j < N; j++)
//            {
//                res[i] += A[j][i] * B[j];
//            }
//        }
          //cache优化
//        for (int i = 0; i < N; i++)
//            res[i] = 0.0;
//        for (int j = 0; j < N; j++)
//            for (int i = 0; i < N; i++)
//                res[i] += A[j][i] * B[j];
          //循环展开
//        for (int i = 0; i < N; i+=4)
//        {
//            res[i] = 0.0;
//            res[i+1] = 0.0;
//            res[i+2] = 0.0;
//            res[i+3] = 0.0;
//        }
//        for (int j = 0; j < N; j+=4)
//            for (int i = 0; i < N; i+=4)
//            {
//                res[i] += A[j][i] * B[j];
//                res[i] += A[j+1][i] * B[j+1];
//                res[i] += A[j+2][i] * B[j+2];
//                res[i] += A[j+3][i] * B[j+3];
//
//                res[i+1] += A[j][i+1] * B[j];
//                res[i+1] += A[j+1][i+1] * B[j+1];
//                res[i+1] += A[j+2][i+1] * B[j+2];
//                res[i+1] += A[j+3][i+1] * B[j+3];
//
//                res[i+2] += A[j][i+2] * B[j];
//                res[i+2] += A[j+1][i+2] * B[j+1];
//                res[i+2] += A[j+2][i+2] * B[j+2];
//                res[i+2] += A[j+3][i+2] * B[j+3];
//
//                res[i+3] += A[j][i+3] * B[j];
//                res[i+3] += A[j+1][i+3] * B[j+1];
//                res[i+3] += A[j+2][i+3] * B[j+2];
//                res[i+3] += A[j+3][i+3] * B[j+3];
//            }
   }


			// end time
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout << "take:" << (tail - head) * 1000.0 / freq
		<< "ms" << endl;
}
