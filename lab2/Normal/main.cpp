#include<iostream>
#include<windows.h>
#define N 2000
using namespace std;

float **m;
float **am;

void m_set()
{
    m=new float*[N];
    for (int i=0; i<N; i++)
    {
        m[i]=new float[N];
    }

    for(int i=0; i<N; i++)
    {
        m[i][i]=1.0;
        for(int j=i+1; j<N; j++)
            m[i][j]=rand()%1000;
    }
    for(int k=0; k<N; k++)
        for(int i=k+1; i<N; i++)
            for(int j=0; j<N; j++)
            {
                m[i][j]+=m[k][j];
                m[i][j]=(int)m[i][j]%1000;
            }
}

void am_set(int bits)
{
    am = (float**)_aligned_malloc(sizeof(float*) * N, bits);
    for (int i = 0; i < N; i++)
    {
        am[i] = (float*)_aligned_malloc(sizeof(float) * N, bits);
    }
    for(int i=0; i<N; i++)
    {
        am[i][i]=1.0;
        for(int j=i+1; j<N; j++)
            am[i][j]=rand()%1000;
    }
    for(int k=0; k<N; k++)
        for(int i=k+1; i<N; i++)
            for(int j=0; j<N; j++)
            {
                am[i][j]+=am[k][j];
                am[i][j]=(int)am[i][j]%1000;
            }

}

void Plain(float **m)
{
    for (int k = 0; k < N; k++)
    {
        for (int j = k + 1; j < N; j++)
        {
            m[k][j] = m[k][j] / m[k][k];
        }
        m[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
        {
            for (int j = k + 1; j < N; j++)
            {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
            m[i][k] = 0;
        }
    }
}

void SSE(float **m)
{
    for (int row = 0; row < N; row++)
    {
        __m128 diagonal = _mm_set1_ps(m[row][row]);
        int col = 0;
        for (col = row + 1; col + 4 <= N; col += 4)
        {
            __m128 row_values = _mm_loadu_ps(&m[row][col]);
            row_values = _mm_div_ps(row_values, diagonal);
            _mm_storeu_ps(&m[row][col], row_values);
        }
        for (; col < N; col++)
        {
            m[row][col] = m[row][col] / m[row][row];
        }
        m[row][row] = 1.0;
        for (int next_row = row + 1; next_row < N; next_row++)
        {
            __m128 pivot_row_value = _mm_set1_ps(m[next_row][row]);
            for (col = row + 1; col + 4 <= N; col += 4)
            {
                __m128 pivot_col_values = _mm_loadu_ps(&m[row][col]);
                __m128 target_values = _mm_loadu_ps(&m[next_row][col]);
                __m128 result_values = _mm_mul_ps(pivot_row_value, pivot_col_values);
                target_values = _mm_sub_ps(target_values, result_values);
                _mm_storeu_ps(&m[next_row][col], target_values);
            }
            for (; col < N; col++)
            {
                m[next_row][col] = m[next_row][col] - m[next_row][row] * m[row][col];
            }
            m[next_row][row] = 0;
        }
    }

}

void Aligin_SSE(float **m)
{
    for (int row = 0; row < N; row++)
    {
        __m128 diagonal = _mm_set1_ps(m[row][row]);
        int col = row + 1;

        for (; col < N && ((intptr_t)(&m[row][col]) % 16) != 0; col++)
        {
            m[row][col] = m[row][col] / m[row][row];
        }

        for (; col + 4 <= N; col += 4)
        {
            __m128 current_values = _mm_load_ps(&m[row][col]);
            current_values = _mm_div_ps(current_values, diagonal);
            _mm_store_ps(&m[row][col], current_values);
        }

        for (; col < N; col++)
        {
            m[row][col] = m[row][col] / m[row][row];
        }

        m[row][row] = 1.0;

        for (int next_row = row + 1; next_row < N; next_row++)
        {
            __m128 pivot_row_value = _mm_set1_ps(m[next_row][row]);
            col = row + 1;

            for (; col < N && ((intptr_t)(&m[row][col]) % 16) != 0; col++)
            {
                m[next_row][col] = m[next_row][col] - m[next_row][row] * m[row][col];
            }

            for (; col + 4 <= N; col += 4)
            {
                __m128 pivot_col_values = _mm_load_ps(&m[row][col]);
                __m128 target_values = _mm_loadu_ps(&m[next_row][col]);
                __m128 result_values = _mm_mul_ps(pivot_row_value, pivot_col_values);
                target_values = _mm_sub_ps(target_values, result_values);
                _mm_storeu_ps(&m[next_row][col], target_values);
            }

            for (; col < N; col++)
            {
                m[next_row][col] = m[next_row][col] - m[next_row][row] * m[row][col];
            }

            m[next_row][row] = 0;
        }
    }
}

void AVX(float **m)
{
    for (int row = 0; row < N; row++)
    {
        __m256 diagonal = _mm256_set1_ps(m[row][row]);
        int col = 0;
        for (col = row + 1; col + 8 <= N; col += 8)
        {
            __m256 row_values = _mm256_loadu_ps(&m[row][col]);
            row_values = _mm256_div_ps(row_values, diagonal);
            _mm256_storeu_ps(&m[row][col], row_values);
        }
        for (; col < N; col++)
        {
            m[row][col] = m[row][col] / m[row][row];
        }
        m[row][row] = 1.0;
        for (int next_row = row + 1; next_row < N; next_row++)
        {
            __m256 pivot_row_value = _mm256_set1_ps(m[next_row][row]);
            for (col = row + 1; col + 8 <= N; col += 8)
            {
                __m256 pivot_col_values = _mm256_loadu_ps(&m[row][col]);
                __m256 target_values = _mm256_loadu_ps(&m[next_row][col]);
                __m256 result_values = _mm256_mul_ps(pivot_row_value, pivot_col_values);
                target_values = _mm256_sub_ps(target_values, result_values);
                _mm256_storeu_ps(&m[next_row][col], target_values);
            }
            for (; col < N; col++)
            {
                m[next_row][col] = m[next_row][col] - m[next_row][row] * m[row][col];
            }
            m[next_row][row] = 0;
        }
    }
}
void Aligin_AVX(float **m)
{
    for (int row = 0; row < N; row++)
    {
        __m256 diagonal = _mm256_set1_ps(m[row][row]);
        int col = row + 1;

        for (; col < N && ((intptr_t)(&m[row][col]) % 32) != 0; col++)
        {
            m[row][col] = m[row][col] / m[row][row];
        }

        for (; col + 8 <= N; col += 8)
        {
            __m256 current_values = _mm256_load_ps(&m[row][col]);
            current_values = _mm256_div_ps(current_values, diagonal);
            _mm256_store_ps(&m[row][col], current_values);
        }

        for (; col < N; col++)
        {
            m[row][col] = m[row][col] / m[row][row];
        }

        m[row][row] = 1.0;

        for (int next_row = row + 1; next_row < N; next_row++)
        {
            __m256 pivot_row_value = _mm256_set1_ps(m[next_row][row]);
            col = row + 1;

            for (; col < N && ((intptr_t)(&m[row][col]) % 32) != 0; col++)
            {
                m[next_row][col] = m[next_row][col] - m[next_row][row] * m[row][col];
            }

            for (; col + 8 <= N; col += 8)
            {
                __m256 pivot_col_values = _mm256_load_ps(&m[row][col]);
                __m256 target_values = _mm256_loadu_ps(&m[next_row][col]);
                __m256 result_values = _mm256_mul_ps(pivot_row_value, pivot_col_values);
                target_values = _mm256_sub_ps(target_values, result_values);
                _mm256_storeu_ps(&m[next_row][col], target_values);
            }

            for (; col < N; col++)
            {
                m[next_row][col] = m[next_row][col] - m[next_row][row] * m[row][col];
            }

            m[next_row][row] = 0;
        }
    }
}

void SSE_first(float **m)
{
    for (int row = 0; row < N; row++)
    {
        for (int j = row + 1; j < N; j++)
        {
            m[row][j] = m[row][j] / m[row][row];
        }
        m[row][row] = 1.0;
        int col=0;
        for (int next_row = row + 1; next_row < N; next_row++)
        {
            __m128 pivot_row_value = _mm_set1_ps(m[next_row][row]);
            for (col = row + 1; col + 4 <= N; col += 4)
            {
                __m128 pivot_col_values = _mm_loadu_ps(&m[row][col]);
                __m128 target_values = _mm_loadu_ps(&m[next_row][col]);
                __m128 result_values = _mm_mul_ps(pivot_row_value, pivot_col_values);
                target_values = _mm_sub_ps(target_values, result_values);
                _mm_storeu_ps(&m[next_row][col], target_values);
            }
            for (; col < N; col++)
            {
                m[next_row][col] = m[next_row][col] - m[next_row][row] * m[row][col];
            }
            m[next_row][row] = 0;
        }
    }
}

void SSE_second(float **m)
{
    for (int row = 0; row < N; row++)
    {
        __m128 diagonal = _mm_set1_ps(m[row][row]);
        int col = 0;
        for (col = row + 1; col + 4 <= N; col += 4)
        {
            __m128 row_values = _mm_loadu_ps(&m[row][col]);
            row_values = _mm_div_ps(row_values, diagonal);
            _mm_storeu_ps(&m[row][col], row_values);
        }
        for (; col < N; col++)
        {
            m[row][col] = m[row][col] / m[row][row];
        }
        m[row][row] = 1.0;
        for (int i = row + 1; i < N; i++)
        {
            for (int j = row + 1; j < N; j++)
            {
                m[i][j] = m[i][j] - m[i][row] * m[row][j];
            }
            m[i][row] = 0;
        }

    }

}

int main()
{

    long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    m_set();
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    Plain(m);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "Plain:" << (tail - head) * 1000 / freq << "ms" << endl;
    for (int i = 0; i < N; i++)
    {
        delete[] m[i];
    }
    delete m;

//    m_set();
//    QueryPerformanceCounter((LARGE_INTEGER*)&head);
//    SSE(m);
//    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
//    cout<<"SSE:"<<(tail-head)*1000/freq<<"ms"<<endl;
//    for (int i = 0; i < N; i++) {
//        delete[] m[i];
//    }
//    delete m;
//
//    am_set(16);
//    QueryPerformanceCounter((LARGE_INTEGER*)&head);
//    Aligin_SSE(am);
//    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
//    cout << "Aligin_SSE:" << (tail - head) * 1000 / freq << "ms" << endl;
//
//    m_set();
//    QueryPerformanceCounter((LARGE_INTEGER*)&head);
//    AVX(m);
//    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
//    cout<<"AVX:"<<(tail-head)*1000/freq<<"ms"<<endl;
//    for (int i = 0; i < N; i++) {
//        delete[] m[i];
//    }
//    delete m;
//
    //am_set(32);
//    QueryPerformanceCounter((LARGE_INTEGER*)&head);
//    Aligin_AVX(am);
//    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
//    cout << "Aligin_AVX:" << (tail - head) * 1000 / freq << "ms" << endl;


//    m_set();
//    QueryPerformanceCounter((LARGE_INTEGER*)&head);
//    SSE_second(m);
//    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
//    cout<<"SSE(first):"<<(tail-head)*1000/freq<<"ms"<<endl;
//    for (int i = 0; i < N; i++) {
//        delete[] m[i];
//    }
//    delete m;
//
//    m_set();
//    QueryPerformanceCounter((LARGE_INTEGER*)&head);
//    SSE_first(m);
//    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
//    cout<<"SSE(second):"<<(tail-head)*1000/freq<<"ms"<<endl;
//    for (int i = 0; i < N; i++) {
//        delete[] m[i];
//    }
//    delete m;




}
