#include <iostream>
#include <windows.h>
#include <stdlib.h>
#include <random>
using namespace std;


void recursion(int n, int a[])
{
    if (n == 1)
        return;
    else
    {
        for (int i = 0; i < n / 2; i++)
            a[i] += a[n - i - 1];
        n = n / 2;
        recursion(n, a);
    }
}
int main()
{
    const int n = 4096;
    int sum=0;
    int a[n];
    for (int i = 0; i < n; i++)
        a[i] = i;
    long long head, tail, freq; // timers
    // similar to CLOCKS_PER_SEC
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    // start time
    QueryPerformanceCounter((LARGE_INTEGER*)&head);

    for (int j = 0; j < 1000; j++)
    {
        //平凡算法
//        for (int i = 0; i < n; i++)
//        {
//            sum += a[i];
//        }

        // 多链路式
//        int sum1 = 0,sum2 = 0;
//        for (long i = 0;i < n;i+=2)
//        {
//            sum1 += a[i];
//            sum2 += a[i + 1];
//        }
//        sum = sum1 + sum2;
        //递归
        //recursion(n, a);
        // 二重循环
//        for (int m = n; m > 1; m /= 2)
//            for (int i = 0; i < m / 2; i++)
//                a[i]= a[i*2] + a[i*2 +1];

    }


    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "take:" << (tail - head) * 1000.0 / freq<< "ms" << endl;


}



