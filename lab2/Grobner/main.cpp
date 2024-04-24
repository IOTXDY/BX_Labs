#include<iostream>
#include<fstream>
#include<windows.h>
#include<string>
#include <sstream>
#include <immintrin.h>
using namespace std;

fstream eliminated_file("F:\\Grobner\\7_8399_6375_4535\\2.txt", ios::in | ios::out);
fstream elimination_file("F:\\Grobner\\7_8399_6375_4535\\1.txt", ios::in | ios::out);

const long n =10000;
const int col = 8399;
const int row_elimination = 6375;
const int row_eliminated = 4535;

int **elimination;
int **eliminated;

void InitEliminated(int row,int **m)
{
    if(!eliminated_file)
    {
        cout<<"fail"<<endl;
        return;
    }
    string str;
    int num;
    for(int i=0; i<row; i++)
    {
        getline(eliminated_file,str);
        stringstream s(str);
        while(s>>num)
        {
            int index=num/32;
            int offset=num%32;
            m[i][index]|=1<<offset;
        }
    }
}

void InitElimination(int row,int **m)
{
    if(!elimination_file)
    {
        cout<<"fail"<<endl;
        return;
    }
    string str;
    int num;
    for(int i=0; i<row; i++)
    {
        getline(elimination_file,str);
        stringstream s(str);
        int term=0;
        s>>num;
        term=num;
        int index=num/32;
        int offset=num%32;
        m[num][index]|=1<<offset;
        while(s>>num)
        {
            index=num/32;
            offset=num%32;
            m[term][index]|=1<<offset;
        }
    }
}

bool IsEmpty(int **m,int maxindex,int row)
{
    for(int index=0; index<maxindex; index++)
        if(m[row][index]!=0)
            return true;
    return false;
}

int GetMaxIndex(int **m,int maxindex,int row)
{
    for(int index=maxindex-1; index>-1; index--)
    {
        if(m[row][index])
        {
            int term=m[row][index];
            for(int b=31; b>-1; b--)
            {
                if(term&(1<<b))
                    return 32*index+b;
            }
        }
    }
    return 0;
}

void Grobner_Plain(int elimination_row,int eliminated_row,int maxindex)
{
    for (int row = 0; row < eliminated_row; row++)
    {
        while (IsEmpty(eliminated, maxindex, row))
        {
            int index = GetMaxIndex(eliminated, maxindex, row);
            if (IsEmpty(elimination, maxindex, index))
            {
                for (int col = 0; col < maxindex; col++)
                {
                    eliminated[row][col] = eliminated[row][col] ^ elimination[index][col];
                }
            }
            else
            {
                for (int col = 0; col < maxindex; col++)
                {
                    elimination[index][col] = eliminated[row][col];
                }
                break;
            }
        }

    }
}

void Grobner_SSE(int elimination_row,int eliminated_row,int maxindex)
{
    for (int row = 0; row < eliminated_row; row++)
    {
        while (IsEmpty(eliminated, maxindex, row))
        {
            int index = GetMaxIndex(eliminated, maxindex, row);
            if (IsEmpty(elimination, maxindex, index))
            {
                int col = 0;
                for (; col + 4 < maxindex; col += 4)
                {
                    __m128i elied = _mm_loadu_si128((__m128i*) & eliminated[row][col]);
                    __m128i eli = _mm_loadu_si128((__m128i*) & elimination[index][col]);
                    __m128i tmp = _mm_xor_si128(elied, eli);
                    _mm_storeu_si128((__m128i*) & eliminated[row][col], tmp);
                }
                for (; col < maxindex; col++)
                {
                    eliminated[row][col] = eliminated[row][col] ^ elimination[index][col];
                }
            }
            else
            {
                int col = 0;
                for (; col + 4 < maxindex; col += 4)
                {
                    __m128i elied = _mm_loadu_si128((__m128i*) & eliminated[row][col]);
                    _mm_storeu_si128((__m128i*) & elimination[index][col], elied);
                }
                for (; col < maxindex; col++)
                {
                    elimination[index][col] = eliminated[row][col];
                }
                break;
            }
        }

    }
}

void Grobner_AVX(int elimination_row,int eliminated_row,int maxindex)
{
    for (int row = 0; row < eliminated_row; row++)
    {
        while (IsEmpty(eliminated, maxindex, row))
        {
            int index = GetMaxIndex(eliminated, maxindex, row);
            if (IsEmpty(elimination, maxindex, index))
            {
                int col = 0;
                for (; col + 8 < maxindex; col += 8)
                {
                    __m256i elied = _mm256_loadu_si256((__m256i*) & eliminated[row][col]);
                    __m256i eli = _mm256_loadu_si256((__m256i*) & elimination[index][col]);
                    __m256i tmp = _mm256_xor_si256(elied, eli);
                    _mm256_storeu_si256((__m256i*) & eliminated[row][col], tmp);
                }
                for (; col < maxindex; col++)
                {
                    eliminated[row][col] = eliminated[row][col] ^ elimination[index][col];
                }
            }
            else
            {
                int col = 0;
                for (; col + 8 < maxindex; col += 8)
                {
                    __m256i elied = _mm256_loadu_si256((__m256i*) & eliminated[row][col]);
                    _mm256_storeu_si256((__m256i*) & elimination[index][col], elied);
                }
                for (; col < maxindex; col++)
                {
                    elimination[index][col] = eliminated[row][col];
                }
                break;
            }
        }

    }
}

int main()
{

    elimination = new int*[n];
    eliminated = new int*[n];

    for (int i = 0; i < n; ++i)
    {
        elimination[i] = new int[n];
        eliminated[i] = new int[n];
    }
    for(int i=0; i<n; i++)
        for(int j=0; j<n; j++)
        {
            eliminated[i][j]=0;
            elimination[i][j]=0;
        }

    int index = (col / 32)+1;
    long long freq, head, tail;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

    InitEliminated(row_eliminated, eliminated);
    InitElimination(row_elimination, elimination);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);

    Grobner_Plain(row_elimination, row_eliminated, index);
    //Grobner_SSE(row_elimination, row_eliminated, index);
    //Grobner_AVX(row_elimination, row_eliminated, index);


    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    double time = (tail - head) * 1000 / freq;
    cout << time << "ms" << endl;
}
