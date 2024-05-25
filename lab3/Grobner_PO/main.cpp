#include<iostream>
#include<fstream>
#include<string>
#include<sstream>
#include<map>
#include<windows.h>
#include<tmmintrin.h>
#include<xmmintrin.h>
#include<emmintrin.h>
#include<pmmintrin.h>
#include<smmintrin.h>
#include<nmmintrin.h>
#include<immintrin.h>
#include<pthread.h>
#include<omp.h>
using namespace std;

#define NUM_THREADS 7


struct threadParam_t {    //�������ݽṹ
	int t_id;
	int num;
};

const int maxsize = 3000;
const int maxrow = 60000; //3000*32>90000 ,����������90000�ı���Ԫ�о���60000��
const int numBasis = 100000;   //���洢90000*100000����Ԫ��

pthread_mutex_t lock;  //д����Ԫ��ʱ��Ҫ����

//long long read = 0;
long long head, tail, freq;

//map<int, int*>iToBasis;    //����Ϊi����Ԫ�ӵ�ӳ��
map<int, int*>ans;			//��

fstream RowFile("E:\\Grobner\\6_3799_2759_1953\\2.txt", ios::in | ios::out);
fstream BasisFile("E:\\Grobner\\6_3799_2759_1953\\1.txt", ios::in | ios::out);


int gRows[maxrow][maxsize];   //����Ԫ�����60000�У�3000��
int gBasis[numBasis][maxsize];  //��Ԫ�����40000�У�3000��

int ifBasis[numBasis] = { 0 };

void reset() {
	//	read = 0;
	memset(gRows, 0, sizeof(gRows));
	memset(gBasis, 0, sizeof(gBasis));
	memset(ifBasis, 0, sizeof(ifBasis));
	RowFile.close();
	BasisFile.close();
	RowFile.open("E:\\Grobner\\6_3799_2759_1953\\2.txt", ios::in | ios::out);
	BasisFile.open("E:\\Grobner\\6_3799_2759_1953\\1.txt", ios::in | ios::out);
	//iToBasis.clear();

	ans.clear();
}

int readBasis() {          //��ȡ��Ԫ��
	for (int i = 0; i < numBasis; i++) {
		if (BasisFile.eof()) {
			//cout << "��ȡ��Ԫ��" << i - 1 << "��" << endl;
			return i - 1;
		}
		string tmp;
		bool flag = false;
		int row = 0;
		getline(BasisFile, tmp);
		stringstream s(tmp);
		int pos;
		while (s >> pos) {
			//cout << pos << " ";
			if (!flag) {
				row = pos;
				flag = true;
				//iToBasis.insert(pair<int, int*>(row, gBasis[row]));
				ifBasis[row] = 1;
			}
			int index = pos / 32;
			int offset = pos % 32;
			gBasis[row][index] = gBasis[row][index] | (1 << offset);
		}
		flag = false;
		row = 0;
	}
}

int readRowsFrom(int pos) {       //��ȡ����Ԫ��
	if (RowFile.is_open())
		RowFile.close();
	RowFile.open("E:\\Grobner\\6_3799_2759_1953\\2.txt", ios::in | ios::out);
	memset(gRows, 0, sizeof(gRows));   //����Ϊ0
	string line;
	for (int i = 0; i < pos; i++) {       //��ȡposǰ���޹���
		getline(RowFile, line);
	}
	for (int i = pos; i < pos + maxrow; i++) {
		int tmp;
		getline(RowFile, line);
		if (line.empty()) {
			//cout << "��ȡ����Ԫ�� " << i << " ��" << endl;
			return i;   //���ض�ȡ������
		}
		bool flag = false;
		stringstream s(line);
		while (s >> tmp) {
			int index = tmp / 32;
			int offset = tmp % 32;
			gRows[i - pos][index] = gRows[i - pos][index] | (1 << offset);
			flag = true;
		}
	}
	cout << "read max rows" << endl;
	return -1;  //�ɹ���ȡmaxrow��

}

int findfirst(int row) {  //Ѱ�ҵ�row�б���Ԫ�е�����
	int first;
	for (int i = maxsize - 1; i >= 0; i--) {
		if (gRows[row][i] == 0)
			continue;
		else {
			int pos = i * 32;
			int offset = 0;
			for (int k = 31; k >= 0; k--) {
				if (gRows[row][i] & (1 << k))
				{
					offset = k;
					break;
				}
			}
			first = pos + offset;
			return first;
		}
	}
	return -1;
}



void writeResult(ofstream& out) {
	for (auto it = ans.rbegin(); it != ans.rend(); it++) {
		int* result = it->second;
		int max = it->first / 32 + 1;
		for (int i = max; i >= 0; i--) {
			if (result[i] == 0)
				continue;
			int pos = i * 32;
			//int offset = 0;
			for (int k = 31; k >= 0; k--) {
				if (result[i] & (1 << k)) {
					out << k + pos << " ";
				}
			}
		}
		out << endl;
	}
}

void GE() {
	int begin = 0;
	int flag;
	flag = readRowsFrom(begin);     //��ȡ����Ԫ��

	int num = (flag == -1) ? maxrow : flag;
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	for (int i = 0; i < num; i++) {
		while (findfirst(i)!= -1) {     //��������
			int first =findfirst(i);      //first������
			if (ifBasis[first]==1) {  //��������Ϊfirst��Ԫ��
				//int* basis = iToBasis.find(first)->second;  //�ҵ�����Ԫ�ӵ�����
				for (int j = 0; j < maxsize; j++) {
					gRows[i][j] = gRows[i][j] ^ gBasis[first][j];     //���������Ԫ

				}
			}
			else {   //����Ϊ��Ԫ��
				//cout << first << endl;
				for (int j = 0; j < maxsize; j++) {
					gBasis[first][j] = gRows[i][j];
				}
				//iToBasis.insert(pair<int, int*>(first, gBasis[first]));
				ifBasis[first] = 1;
				ans.insert(pair<int, int*>(first, gBasis[first]));
				break;
			}
		}
	}
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout << "Ordinary time:" << (tail - head) * 1000 / freq << "ms" << endl;
}

void GE_omp() {
	int begin = 0;
	int flag;
	flag = readRowsFrom(begin);     //��ȡ����Ԫ��
	//int i = 0, j = 0;
	int t_id = omp_get_thread_num();
	int num = (flag == -1) ? maxrow : flag;
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
#pragma omp parallel num_threads(NUM_THREADS)
	{
#pragma omp for schedule(guided)
	for (int i = 0; i < num; i++) {
		while (findfirst(i) != -1) {     //��������
			int first = findfirst(i);      //first������
			if (ifBasis[first] == 1) {  //��������Ϊfirst��Ԫ��
				for (int j = 0; j < maxsize; j++) {
					gRows[i][j] = gRows[i][j] ^ gBasis[first][j];     //���������Ԫ

				}
			}
			else {   //����Ϊ��Ԫ��
#pragma omp critical
				if (ifBasis[first] == 0) {
					for (int j = 0; j < maxsize; j++) {
						gBasis[first][j] = gRows[i][j];
					}
					//iToBasis.insert(pair<int, int*>(first, gBasis[first]));
					ifBasis[first] = 1;
					ans.insert(pair<int, int*>(first, gBasis[first]));
				}
			}

		}
	}
	}
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout << "Omp time:" << (tail - head) * 1000 / freq << "ms" << endl;
}


void AVX_GE() {
	int begin = 0;
	int flag;
	flag = readRowsFrom(begin);     //��ȡ����Ԫ��
	int num = (flag == -1) ? maxrow : flag;
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	for (int i = 0; i < num; i++) {
		while (findfirst(i) != -1) {
			int first = findfirst(i);
			if (ifBasis[first]==1) {  //���ڸ���Ԫ��
				//int* basis = iToBasis.find(first)->second;
				int j = 0;
				for (; j + 8 < maxsize; j += 8) {
					__m256i vij = _mm256_loadu_si256((__m256i*) & gRows[i][j]);
					__m256i vj = _mm256_loadu_si256((__m256i*) & gBasis[first][j]);
					__m256i vx = _mm256_xor_si256(vij, vj);
					_mm256_storeu_si256((__m256i*) & gRows[i][j], vx);
				}
				for (; j < maxsize; j++) {
					gRows[i][j] = gRows[i][j] ^ gBasis[first][j];
				}
			}
			else {
				int j = 0;
				for (; j + 8 < maxsize; j += 8) {
					__m256i vij = _mm256_loadu_si256((__m256i*) & gRows[i][j]);
					_mm256_storeu_si256((__m256i*) & gBasis[first][j], vij);
				}
				for (; j < maxsize; j++) {
					gBasis[first][j] = gRows[i][j];
				}
				//iToBasis.insert(pair<int, int*>(first, gBasis[first]));
				ifBasis[first] = 1;
				ans.insert(pair<int, int*>(first, gBasis[first]));
				break;

			}
		}
	}
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout << "AVX time:" << (tail - head) * 1000 / freq << "ms" << endl;
}

void AVX_GE_omp() {
	int begin = 0;
	int flag;
	flag = readRowsFrom(begin);     //��ȡ����Ԫ��
	int num = (flag == -1) ? maxrow : flag;
	int i = 0, j = 0;
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
#pragma omp parallel  num_threads(NUM_THREADS),private(i,j)
#pragma omp for schedule(guided)
	for (i = 0; i < num; i++) {
		while (findfirst(i) != -1) {
			int first = findfirst(i);
			if (ifBasis[first]==1) {  //���ڸ���Ԫ��
				//int* basis = iToBasis.find(first)->second;
				j = 0;
				for (; j + 8 < maxsize; j += 8) {
					__m256i vij = _mm256_loadu_si256((__m256i*) & gRows[i][j]);
					__m256i vj = _mm256_loadu_si256((__m256i*) & gBasis[first][j]);
					__m256i vx = _mm256_xor_si256(vij, vj);
					_mm256_storeu_si256((__m256i*) & gRows[i][j], vx);
				}
				for (; j < maxsize; j++) {
					gRows[i][j] = gRows[i][j] ^ gBasis[first][j];
				}
			}
			else {
#pragma omp critical
				{j = 0;
				for (; j + 8 < maxsize; j += 8) {
					__m256i vij = _mm256_loadu_si256((__m256i*) & gRows[i][j]);
					_mm256_storeu_si256((__m256i*) & gBasis[first][j], vij);
				}
				for (; j < maxsize; j++) {
					gBasis[first][j] = gRows[i][j];
				}
				//iToBasis.insert(pair<int, int*>(first, gBasis[first]));
				ifBasis[first] = 1;
				ans.insert(pair<int, int*>(first, gBasis[first]));
				}
			}
		}
	}
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout << "AVX_omp time:" << (tail - head) * 1000 / freq << "ms" << endl;
}

void* GE_lock_thread(void* param) {
	threadParam_t* p = (threadParam_t*)param;
	int t_id = p->t_id;
	int num = p->num;
	for (int i = t_id; i < num; i += NUM_THREADS) {
		while (findfirst(i) != -1) {
			int first = findfirst(i);      //first������
			if (ifBasis[first]==1) {  //��������Ϊfirst��Ԫ��
				//int* basis = iToBasis.find(first)->second;  //�ҵ�����Ԫ�ӵ�����
				for (int j = 0; j < maxsize; j++) {
					gRows[i][j] = gRows[i][j] ^ gBasis[first][j];     //���������Ԫ

				}
			}
			else {   //����Ϊ��Ԫ��
				pthread_mutex_lock(&lock); //�����first����Ԫ��û�б�ռ�ã������
				if (ifBasis[first]==1)
				{
					pthread_mutex_unlock(&lock);
					continue;
				}

				for (int j = 0; j < maxsize; j++) {
					gBasis[first][j] = gRows[i][j];     //��Ԫ�ӵ�д��
				}
				//iToBasis.insert(pair<int, int*>(first, gBasis[first]));
				ifBasis[first] = 1;
				ans.insert(pair<int, int*>(first, gBasis[first]));
				pthread_mutex_unlock(&lock);          //����
				break;
			}

		}
	}
	pthread_exit(NULL);
	return NULL;
}

void GE_pthread() {
	int begin = 0;
	int flag;
	flag = readRowsFrom(begin);     //��ȡ����Ԫ��

	int num = (flag == -1) ? maxrow : flag;

	pthread_mutex_init(&lock, NULL);  //��ʼ����

	pthread_t* handle = (pthread_t*)malloc(NUM_THREADS * sizeof(pthread_t));
	threadParam_t* param = (threadParam_t*)malloc(NUM_THREADS * sizeof(threadParam_t));

	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	for (int t_id = 0; t_id < NUM_THREADS; t_id++) {//��������
		param[t_id].t_id = t_id;
		param[t_id].num = num;
		pthread_create(&handle[t_id], NULL, GE_lock_thread, &param[t_id]);
	}

	for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
		pthread_join(handle[t_id], NULL);
	}

	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout << "GE_pthread time:" << (tail - head) * 1000 / freq << "ms" << endl;
	free(handle);
	free(param);
	pthread_mutex_destroy(&lock);
}

void* AVX_lock_thread(void* param) {

	threadParam_t* p = (threadParam_t*)param;
	int t_id = p->t_id;
	int num = p->num;

	for (int i = t_id; i  < num; i += NUM_THREADS) {
		while (findfirst(i) != -1) {
			int first = findfirst(i);
			if (ifBasis[first]==1) {  //���ڸ���Ԫ��
				//int* basis = iToBasis.find(first)->second;
				int j = 0;
				for (; j + 8 < maxsize; j += 8) {
					__m256i vij = _mm256_loadu_si256((__m256i*) & gRows[i][j]);
					__m256i vj = _mm256_loadu_si256((__m256i*) & gBasis[first][j]);
					__m256i vx = _mm256_xor_si256(vij, vj);
					_mm256_storeu_si256((__m256i*) & gRows[i][j], vx);
				}
				for (; j < maxsize; j++) {
					gRows[i][j] = gRows[i][j] ^ gBasis[first][j];
				}
			}
			else {
				pthread_mutex_lock(&lock); //�����first����Ԫ��û�б�ռ�ã������
				if (ifBasis[first]==1)
				{
					pthread_mutex_unlock(&lock);
					continue;
				}
				int j = 0;
				for (; j + 8 < maxsize; j += 8) {
					__m256i vij = _mm256_loadu_si256((__m256i*) & gRows[i][j]);
					_mm256_storeu_si256((__m256i*) & gBasis[first][j], vij);
				}
				for (; j < maxsize; j++) {
					gBasis[first][j] = gRows[i][j];
				}
				//iToBasis.insert(pair<int, int*>(first, gBasis[first]));
				ifBasis[first] = 1;
				ans.insert(pair<int, int*>(first, gBasis[first]));
				pthread_mutex_unlock(&lock);
				break;
			}
		}
	}
	pthread_exit(NULL);
	return NULL;
}

void AVX_pthread() {
	int begin = 0;
	int flag;
	flag = readRowsFrom(begin);     //��ȡ����Ԫ��

	int num = (flag == -1) ? maxrow : flag;

	pthread_mutex_init(&lock, NULL);  //��ʼ����

	pthread_t* handle = (pthread_t*)malloc(NUM_THREADS * sizeof(pthread_t));
	threadParam_t* param = (threadParam_t*)malloc(NUM_THREADS * sizeof(threadParam_t));

	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	for (int t_id = 0; t_id < NUM_THREADS; t_id++) {//��������
		param[t_id].t_id = t_id;
		param[t_id].num = num;
		pthread_create(&handle[t_id], NULL, AVX_lock_thread, &param[t_id]);
	}

	for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
		pthread_join(handle[t_id], NULL);
	}

	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout << "AVX_pthread time:" << (tail - head) * 1000 / freq << "ms" << endl;
	free(handle);
	free(param);
	pthread_mutex_destroy(&lock);
}

int main() {

		QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

//		readBasis();
//		GE();
//		//writeResult(out);
//
//		reset();

//		readBasis();
//		GE_omp();
//		//writeResult(out4);
//
//		reset();
//
//		readBasis();
//		AVX_GE_omp();
//		//writeResult(out5);
//
//		reset();
//
//		readBasis();
//		AVX_GE();
//		//writeResult(out1);
//
//		reset();
//
//		readBasis();
//		GE_pthread();
//		//writeResult(out2);
//
//		reset();
//
		readBasis();
		AVX_pthread();
		//writeResult(out3);

		reset();
}