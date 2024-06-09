
#include<iostream>
#include<fstream>
#include<string>
#include<sstream>
#include<map>
#include<vector>
#include<windows.h>
#include<tmmintrin.h>
#include<xmmintrin.h>
#include<emmintrin.h>
#include<pmmintrin.h>
#include<smmintrin.h>
#include<nmmintrin.h>
#include<immintrin.h>
#include<mpi.h>
#include<omp.h>
using namespace std;

#define NUM_THREADS 8


const int maxsize = 3000;
const int maxrow = 40000; 
const int numBasis = 40000;   
int num;

vector<int> tmpAns;

//long long read = 0;
long long head, tail, freq;


map<int, int*>ans;			

fstream RowFile("F:\\Grobner\\3_562_170_53\\2.txt", ios::in | ios::out);
fstream BasisFile("F:\\Grobner\\3_562_170_53\\1.txt", ios::in | ios::out);

ofstream out_mpi("��Ԫ���(MPI).txt");

int gRows[maxrow][maxsize];   
int gBasis[numBasis][maxsize];  
int answers[maxrow][maxsize]; 
map<int, int>firstToRow; 

int ifBasis[numBasis] = { 0 };
int ifDone[maxrow] = { 0 };

void reset() {
	//	read = 0;
	memset(gRows, 0, sizeof(gRows));
	memset(gBasis, 0, sizeof(gBasis));
	memset(ifBasis, 0, sizeof(ifBasis));
	RowFile.close();
	BasisFile.close();
	RowFile.open("F:\\Grobner\\3_562_170_53\\2.txt", ios::in | ios::out);
	BasisFile.open("F:\\Grobner\\3_562_170_53\\1.txt", ios::in | ios::out);
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

				//cout << row << endl;
			}
			int index = pos / 32;
			int offset = pos % 32;
			gBasis[row][index] = gBasis[row][index] | (1 << offset);
		}
		flag = false;
		row = 0;
	}
}

int readRowsFrom(int pos) {      
	if (RowFile.is_open())
		RowFile.close();
	RowFile.open("F:\\Grobner\\3_562_170_53\\2.txt", ios::in | ios::out);
	memset(gRows, 0, sizeof(gRows));   
	string line;
	for (int i = 0; i < pos; i++) {       
		getline(RowFile, line);
	}
	for (int i = pos; i < pos + maxrow; i++) {
		int tmp;
		getline(RowFile, line);
		if (line.empty()) {
			//cout << "��ȡ����Ԫ�� " << i << " ��" << endl;
			return i;   
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
	return -1; 

}

int findfirst(int row) {  
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

int _findfirst(int row) {  
	int first;
	for (int i = maxsize - 1; i >= 0; i--) {
		if (answers[row][i] == 0)
			continue;
		else {
			int pos = i * 32;
			int offset = 0;
			for (int k = 31; k >= 0; k--) {
				if (answers[row][i] & (1 << k))
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
	flag = readRowsFrom(begin);    

	int num = (flag == -1) ? maxrow : flag;
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	for (int i = 0; i < num; i++) {
		while (findfirst(i) != -1) {    
			int first = findfirst(i);      
			if (ifBasis[first] == 1) {  
				//int* basis = iToBasis.find(first)->second;  
				for (int j = 0; j < maxsize; j++) {
					gRows[i][j] = gRows[i][j] ^ gBasis[first][j];     

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
	flag = readRowsFrom(begin);     
	//int i = 0, j = 0;
	int t_id = omp_get_thread_num();
	int num = (flag == -1) ? maxrow : flag;
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
#pragma omp parallel num_threads(NUM_THREADS)
	{
#pragma omp for schedule(guided)
		for (int i = 0; i < num; i++) {
			//cout << omp_get_thread_num() << "�߳�" << endl;
			while (findfirst(i) != -1) {     
				int first = findfirst(i);      
				if (ifBasis[first] == 1) {  
					//cout << first << "from" << omp_get_thread_num() << endl;
					for (int j = 0; j < maxsize; j++) {
						gRows[i][j] = gRows[i][j] ^ gBasis[first][j];    

					}
				}
				else {   
					//cout << first <<"from"<< omp_get_thread_num()<< endl;
#pragma omp critical
					if (ifBasis[first] == 0) {
						for (int j = 0; j < maxsize; j++) {
							gBasis[first][j] = gRows[i][j];
						}
						//iToBasis.insert(pair<int, int*>(first, gBasis[first]));
						ifBasis[first] = 1;
						ans.insert(pair<int, int*>(first, gBasis[first]));
					}
					//break;    
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
			if (ifBasis[first] == 1) {  //���ڸ���Ԫ��
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
#pragma omp parallel num_threads(NUM_THREADS),private(i,j)
#pragma omp for ordered schedule(guided)
	for (i = 0; i < num; i++) {
		int first = findfirst(i);
		while (first != -1) {
			//first = findfirst(i);
			if (ifBasis[first] == 1) {  //���ڸ���Ԫ��
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
				first = findfirst(i);
			}
			else {
#pragma omp ordered
				{
					while (ifBasis[first] == 1 && first != -1) {  //���ڸ���Ԫ��
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
						first = findfirst(i);
					}

					j = 0;
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
				first = -1;
			}
		}
	}
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout << "AVX_omp time:" << (tail - head) * 1000 / freq << "ms" << endl;
}

void writeResult_MPI(ofstream& out) { //mpi�汾
	for (int j = 0; j < num; j++) {
		for (int i = maxsize - 1; i >= 0; i--) {
			if (answers[j][i] == 0)
				continue;
			int pos = i * 32;
			//int offset = 0;
			for (int k = 31; k >= 0; k--) {
				if (answers[j][i] & (1 << k)) {
					out << k + pos << " ";
				}
			}
		}
		out << endl;
	}
}

void GE_MPI(int argc, char* argv[]) {
	int flag;
	double start_time = 0;
	double end_time = 0;
	MPI_Init(&argc, &argv);
	int total = 0;
	int rank = 0;
	int i = 0;
	int j = 0;
	int begin = 0, end = 0;
	MPI_Status status;
	MPI_Comm_size(MPI_COMM_WORLD, &total);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (rank == 0) {
		flag = readRowsFrom(0);     //��ȡ����Ԫ��
		num = (flag == -1) ? maxrow : flag;
		begin = rank * num / total;
		end = (rank == total - 1) ? num : (rank + 1) * (num / total);
		for (i = 1; i < total; i++) {
			MPI_Send(&num, 1, MPI_INT, i, 0, MPI_COMM_WORLD);//0�Ǳ���Ԫ������
			int b = i * (num / total);
			int e = (i == total - 1) ? num : (i + 1) * (num / total);
			for (j = b; j < e; j++) {
				MPI_Send(&gRows[j][0], maxsize, MPI_INT, i, 1, MPI_COMM_WORLD);//1ʱ����Ԫ������
			}
		}

	}
	else {
		//cout << rank << " recving" << endl;
		MPI_Recv(&num, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
		//cout << rank << " num:" << num << endl;
		begin = rank * (num / total);
		end = (rank == total - 1) ? num : (rank + 1) * (num / total);
		for (i = begin; i < end; i++) {
			MPI_Recv(&gRows[i][0], maxsize, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
		}
	}

	//cout << rank << "  " << begin << "   " << end << endl << endl;

	MPI_Barrier(MPI_COMM_WORLD);  //��ʱÿ�����̶��õ�������
	start_time = MPI_Wtime();
	for (i = begin; i < end; i++) {
		int first = findfirst(i);

		while (first != -1) {     //δ��Ԫ��ϣ���������

			//int first = findfirst(i);      //first������

			if (ifBasis[first] == 1) {  //��������Ϊfirst��Ԫ��
				for (j = 0; j < maxsize; j++) {
					gRows[i][j] = gRows[i][j] ^ gBasis[first][j];     //���������Ԫ

				}
				first = findfirst(i);
			}
			else {   //����Ϊ��Ԫ��
				tmpAns.push_back(first);
				//ifDone[i] = 1;
				if (rank == 0) {
					for (j = 0; j < maxsize; j++) {
						gBasis[first][j] = gRows[i][j];
						answers[i][j] = gRows[i][j];
					}
					ifBasis[first] = 1;  //������0�Ž�����Ԫ����
				}
				//ifBasis[first] = 1;
				break;
			}
		}
		if (first == -1)
			tmpAns.push_back(-1);
	}
	//cout << rank << " done for own" << endl;
	for (i = 0; i < rank; i++) {

		int b = i * (num / total);
		int e = b + num / total;
		//cout << rank << " wating " << i << endl;
		for (j = b; j < e; j++) {
			MPI_Recv(&answers[j][0], maxsize, MPI_INT, i, 2, MPI_COMM_WORLD, &status);//�������Խ���i����Ԫ�����������Ϊ֮�����Ԫ��
			int first = _findfirst(j);
			firstToRow.insert(pair<int, int>(first, j));//��¼��������Ϣ
		}
		for (j = begin; j < end; j++) {  //��0����Ҫ���ж�����Ԫ���Դ�ǰ���̵Ľ����Ϊ��Ԫ��
			//cout << rank << " doing " << j << endl;

			int first = tmpAns.at(j - begin);
			if (first == -1)
				continue;

			while ((firstToRow.find(first) != firstToRow.end() || ifBasis[first] == 1) && first != -1) {  //���ڿ���Ԫ��
				if (firstToRow.find(first) != firstToRow.end()) {  //��Ԫ�������Ԫ��
					int row = firstToRow.find(first)->second;
					for (int k = 0; k < maxsize; k++) {
						gRows[j][k] = gRows[j][k] ^ answers[row][k];
					}
					first = findfirst(i);
					//cout << "done 1 :" << first << endl;
				}

				if (first == -1)
					break;
				if (ifBasis[first] == 1) {
					for (int k = 0; k < maxsize; k++) {
						gRows[j][k] = gRows[j][k] ^ gBasis[first][k];     //���������Ԫ

					}
					first = findfirst(i);
				}
			}

		}
		//cout << rank << "done at " << i << endl;
	}
	if (rank != 0) {
		for (i = begin; i < end; i++) {
			int first = findfirst(i);
			if (first == -1)
				continue;
			//cout << rank << " doing " << first << endl;
			while ((firstToRow.find(first) != firstToRow.end() || ifBasis[first] == 1) && first != -1) {  //���ڿ���Ԫ��
				if (firstToRow.find(first) != firstToRow.end()) {  //��Ԫ�������Ԫ��
					int row = firstToRow.find(first)->second;
					for (int k = 0; k < maxsize; k++) {
						gRows[i][k] = gRows[i][k] ^ answers[row][k];
					}
					first = findfirst(i);
					//cout << "done 1 :" << first << endl;
				}

				if (first == -1)
					break;
				if (ifBasis[first] == 1) {
					for (int k = 0; k < maxsize; k++) {
						gRows[i][k] = gRows[i][k] ^ gBasis[first][k];     //���������Ԫ

					}
					first = findfirst(i);
				}
			}
			for (j = 0; j < maxsize; j++) {
				gBasis[first][j] = gRows[i][j];
				answers[i][j] = gRows[i][j];  //�������̵���Ԫ����������firstToRow
			}
			ifBasis[first] = 1;


		}

	}
	//cout << rank << " done process!!!!" << endl << endl;
	for (i = rank + 1; i < total; i++) {
		for (j = begin; j < end; j++) {

			MPI_Send(&answers[j][0], maxsize, MPI_INT, i, 2, MPI_COMM_WORLD);//2�Ǹý��̵���Ԫ�����������Ϊ֮����̵���Ԫ��
		}
	}

	if (rank == total - 1) {
		end_time = MPI_Wtime();
		cout << "MPI�Ż��汾��ʱ�� " << 1000 * (end_time - start_time) << "ms" << endl;
		writeResult_MPI(out_mpi);
		out_mpi.close();
	}
	MPI_Finalize();

}
void GE_MPI_omp(int argc, char* argv[]) {
	int flag;
	double start_time = 0;
	double end_time = 0;
	MPI_Init(&argc, &argv);
	int total = 0;
	int rank = 0;
	int i = 0;
	int j = 0;
	int begin = 0, end = 0;
	MPI_Status status;
	MPI_Comm_size(MPI_COMM_WORLD, &total);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (rank == 0) {
		flag = readRowsFrom(0);     //��ȡ����Ԫ��
		num = (flag == -1) ? maxrow : flag;
		begin = rank * num / total;
		end = (rank == total - 1) ? num : (rank + 1) * (num / total);
		for (i = 1; i < total; i++) {
			MPI_Send(&num, 1, MPI_INT, i, 0, MPI_COMM_WORLD);//0�Ǳ���Ԫ������
			int b = i * (num / total);
			int e = (i == total - 1) ? num : (i + 1) * (num / total);
			for (j = b; j < e; j++) {
				MPI_Send(&gRows[j][0], maxsize, MPI_INT, i, 1, MPI_COMM_WORLD);//1ʱ����Ԫ������
			}
		}

	}
	else {
		//cout << rank << " recving" << endl;
		MPI_Recv(&num, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
		//cout << rank << " num:" << num << endl;
		begin = rank * (num / total);
		end = (rank == total - 1) ? num : (rank + 1) * (num / total);
		for (i = begin; i < end; i++) {
			MPI_Recv(&gRows[i][0], maxsize, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
		}
	}

	//cout << rank << "  " << begin << "   " << end << endl << endl;

	MPI_Barrier(MPI_COMM_WORLD);  //��ʱÿ�����̶��õ�������
	start_time = MPI_Wtime();
#pragma omp parallel  num_threads(NUM_THREADS),private(i,j)
#pragma omp for ordered schedule(guided)
	for (i = begin; i < end; i++) {
		int first = findfirst(i);

		while (first != -1) {     //δ��Ԫ��ϣ���������

			//int first = findfirst(i);      //first������

			if (ifBasis[first] == 1) {  //��������Ϊfirst��Ԫ��
				for (j = 0; j < maxsize; j++) {
					gRows[i][j] = gRows[i][j] ^ gBasis[first][j];     //���������Ԫ

				}
				first = findfirst(i);
			}
			else {   //����Ϊ��Ԫ��
				//ifDone[i] = 1;
				if (rank == 0) {
#pragma omp ordered
					for (j = 0; j < maxsize; j++) {
						gBasis[first][j] = gRows[i][j];
						answers[i][j] = gRows[i][j];
					}
					ifBasis[first] = 1;  //������0�Ž�����Ԫ����
				}
				//ifBasis[first] = 1;
				first = -1;
			}
		}

	}
	//cout << rank << " done for own" << endl;
	for (i = 0; i < rank; i++) {

		int b = i * (num / total);
		int e = b + num / total;
		//cout << rank << " wating " << i << endl;
		for (j = b; j < e; j++) {
			MPI_Recv(&answers[j][0], maxsize, MPI_INT, i, 2, MPI_COMM_WORLD, &status);//�������Խ���i����Ԫ�����������Ϊ֮�����Ԫ��
			int first = _findfirst(j);
			firstToRow.insert(pair<int, int>(first, j));//��¼��������Ϣ
		}
#pragma omp for schedule(guided)
		for (j = begin; j < end; j++) {  //��0����Ҫ���ж�����Ԫ���Դ�ǰ���̵Ľ����Ϊ��Ԫ��
			//cout << rank << " doing " << j << endl;

			int first = findfirst(j);

			while ((firstToRow.find(first) != firstToRow.end() || ifBasis[first] == 1) && first != -1) {  //���ڿ���Ԫ��
				if (firstToRow.find(first) != firstToRow.end()) {  //��Ԫ�������Ԫ��
					int row = firstToRow.find(first)->second;
					for (int k = 0; k < maxsize; k++) {
						gRows[j][k] = gRows[j][k] ^ answers[row][k];
					}
					first = findfirst(i);
					//cout << "done 1 :" << first << endl;
				}


				if (ifBasis[first] == 1 && first != -1) {
					for (int k = 0; k < maxsize; k++) {
						gRows[j][k] = gRows[j][k] ^ gBasis[first][k];     //���������Ԫ

					}
					first = findfirst(i);
				}
			}

		}
		//cout << rank << "done at " << i << endl;
	}
	if (rank != 0) {
		for (i = begin; i < end; i++) {
			int first = findfirst(i);

			//cout << rank << " doing " << first << endl;
			while ((firstToRow.find(first) != firstToRow.end() || ifBasis[first] == 1) && first != -1) {  //���ڿ���Ԫ��
				if (firstToRow.find(first) != firstToRow.end()) {  //��Ԫ�������Ԫ��
					int row = firstToRow.find(first)->second;
					for (int k = 0; k < maxsize; k++) {
						gRows[i][k] = gRows[i][k] ^ answers[row][k];
					}
					first = findfirst(i);
					//cout << "done 1 :" << first << endl;
				}

				if (ifBasis[first] == 1 && first != -1) {
					for (int k = 0; k < maxsize; k++) {
						gRows[i][k] = gRows[i][k] ^ gBasis[first][k];     //���������Ԫ

					}
					first = findfirst(i);
				}
			}
			for (j = 0; j < maxsize; j++) {
				gBasis[first][j] = gRows[i][j];
				answers[i][j] = gRows[i][j];  //�������̵���Ԫ����������firstToRow
			}
			ifBasis[first] = 1;


		}

	}
	//cout << rank << " done process!!!!" << endl << endl;
	for (i = rank + 1; i < total; i++) {
		for (j = begin; j < end; j++) {

			MPI_Send(&answers[j][0], maxsize, MPI_INT, i, 2, MPI_COMM_WORLD);//2�Ǹý��̵���Ԫ�����������Ϊ֮����̵���Ԫ��
		}
	}

	if (rank == total - 1) {
		end_time = MPI_Wtime();
		cout << "MPI�Ż��汾��ʱ�� " << 1000 * (end_time - start_time) << "ms" << endl;
		writeResult_MPI(out_mpi);
		out_mpi.close();
	}
	MPI_Finalize();

}

void GE_MPI_AVX(int argc, char* argv[]) {
	int flag;
	double start_time = 0;
	double end_time = 0;
	MPI_Init(&argc, &argv);
	int total = 0;
	int rank = 0;
	int i = 0;
	int j = 0;
	int begin = 0, end = 0;
	MPI_Status status;
	MPI_Comm_size(MPI_COMM_WORLD, &total);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (rank == 0) {
		flag = readRowsFrom(0);     //��ȡ����Ԫ��
		num = (flag == -1) ? maxrow : flag;
		begin = rank * num / total;
		end = (rank == total - 1) ? num : (rank + 1) * (num / total);
		for (i = 1; i < total; i++) {
			MPI_Send(&num, 1, MPI_INT, i, 0, MPI_COMM_WORLD);//0�Ǳ���Ԫ������
			int b = i * (num / total);
			int e = (i == total - 1) ? num : (i + 1) * (num / total);
			for (j = b; j < e; j++) {
				MPI_Send(&gRows[j][0], maxsize, MPI_INT, i, 1, MPI_COMM_WORLD);//1ʱ����Ԫ������
			}
		}

	}
	else {
		//cout << rank << " recving" << endl;
		MPI_Recv(&num, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
		//cout << rank << " num:" << num << endl;
		begin = rank * (num / total);
		end = (rank == total - 1) ? num : (rank + 1) * (num / total);
		for (i = begin; i < end; i++) {
			MPI_Recv(&gRows[i][0], maxsize, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
		}
	}



	MPI_Barrier(MPI_COMM_WORLD);  //��ʱÿ�����̶��õ�������
	start_time = MPI_Wtime();

	for (i = begin; i < end; i++) {
		int first = findfirst(i);
		while (first != -1) {     //δ��Ԫ��ϣ���������

			//int first = findfirst(i);      //first������

			if (ifBasis[first] == 1) {  //��������Ϊfirst��Ԫ��
				for (j = 0; j + 8 < maxsize; j += 8) {
					__m256i vij = _mm256_loadu_si256((__m256i*) & gRows[i][j]);
					__m256i vj = _mm256_loadu_si256((__m256i*) & gBasis[first][j]);
					__m256i vx = _mm256_xor_si256(vij, vj);
					_mm256_storeu_si256((__m256i*) & gRows[i][j], vx);
				}
				for (; j < maxsize; j++) {
					gRows[i][j] = gRows[i][j] ^ gBasis[first][j];
				}
				first = findfirst(i);
			}
			else {   //����Ϊ��Ԫ��
				//tmpAns.push_back(first);
				//ifDone[i] = 1;
//#pragma omp ordered
				if (rank == 0) {
					for (j = 0; j + 8 < maxsize; j += 8) {
						__m256i vij = _mm256_loadu_si256((__m256i*) & gRows[i][j]);
						_mm256_storeu_si256((__m256i*) & gBasis[first][j], vij);
						_mm256_storeu_si256((__m256i*) & answers[i][j], vij);
					}
					for (; j < maxsize; j++) {
						gBasis[first][j] = gRows[i][j];
						answers[i][j] = gRows[i][j];
					}
					ifBasis[first] = 1;  //������0�Ž�����Ԫ����
				}
				//ifBasis[first] = 1;
				break;
			}
		}
	}
	//cout << rank << " done for own" << endl;
	for (i = 0; i < rank; i++) {

		int b = i * (num / total);
		int e = b + num / total;
		//cout << rank << " wating " << i << endl;
		for (j = b; j < e; j++) {
			MPI_Recv(&answers[j][0], maxsize, MPI_INT, i, 2, MPI_COMM_WORLD, &status);//�������Խ���i����Ԫ�����������Ϊ֮�����Ԫ��
			int first = _findfirst(j);
			firstToRow.insert(pair<int, int>(first, j));//��¼��������Ϣ
		}
		//#pragma omp for schedule(guided)
		for (j = begin; j < end; j++) {  //��0����Ҫ���ж�����Ԫ���Դ�ǰ���̵Ľ����Ϊ��Ԫ��
			//cout << rank << " doing " << j << endl;

			int first = findfirst(j);
			while ((firstToRow.find(first) != firstToRow.end() || ifBasis[first] == 1) && first != -1) {  //���ڿ���Ԫ��
				if (firstToRow.find(first) != firstToRow.end()) {  //��Ԫ�������Ԫ��
					int row = firstToRow.find(first)->second;
					int k = 0;
					for (k = 0; k + 8 < maxsize; k += 8) {
						__m256i vij = _mm256_loadu_si256((__m256i*) & gRows[j][k]);
						__m256i vj = _mm256_loadu_si256((__m256i*) & answers[row][k]);
						__m256i vx = _mm256_xor_si256(vij, vj);
						_mm256_storeu_si256((__m256i*) & gRows[j][k], vx);
					}
					for (; k < maxsize; k++) {
						gRows[j][k] = gRows[j][k] ^ answers[row][k];
					}
					first = findfirst(i);
					//cout << "done 1 :" << first << endl;
				}
				if (ifBasis[first] == 1) {
					int k = 0;
					for (k = 0; k + 8 < maxsize; k += 8) {
						__m256i vij = _mm256_loadu_si256((__m256i*) & gRows[j][k]);
						__m256i vj = _mm256_loadu_si256((__m256i*) & gBasis[first][k]);
						__m256i vx = _mm256_xor_si256(vij, vj);
						_mm256_storeu_si256((__m256i*) & gRows[j][k], vx);
					}
					for (; k < maxsize; k++) {
						gRows[j][k] = gRows[j][k] ^ gBasis[first][k];
					}
					first = findfirst(i);
				}
			}

		}
	}

	if (rank != 0) {
		for (i = begin; i < end; i++) {
			int first = findfirst(i);
			if (first == -1)
				continue;
			//cout << rank << " doing " << first << endl;
			while ((firstToRow.find(first) != firstToRow.end() || ifBasis[first] == 1) && first != -1) {  //���ڿ���Ԫ��
				if (firstToRow.find(first) != firstToRow.end()) {  //��Ԫ�������Ԫ��
					int row = firstToRow.find(first)->second;
					int k = 0;
					for (k = 0; k + 8 < maxsize; k += 8) {
						__m256i vij = _mm256_loadu_si256((__m256i*) & gRows[i][k]);
						__m256i vj = _mm256_loadu_si256((__m256i*) & answers[row][k]);
						__m256i vx = _mm256_xor_si256(vij, vj);
						_mm256_storeu_si256((__m256i*) & gRows[i][k], vx);
					}
					for (; k < maxsize; k++) {
						gRows[i][k] = gRows[i][k] ^ answers[row][k];
					}
					first = findfirst(i);
					//cout << "done 1 :" << first << endl;
				}

				if (first == -1)
					break;
				if (ifBasis[first] == 1) {
					int k = 0;
					for (k = 0; k + 8 < maxsize; k += 8) {
						__m256i vij = _mm256_loadu_si256((__m256i*) & gRows[i][k]);
						__m256i vj = _mm256_loadu_si256((__m256i*) & gBasis[first][k]);
						__m256i vx = _mm256_xor_si256(vij, vj);
						_mm256_storeu_si256((__m256i*) & gRows[i][k], vx);
					}
					for (; k < maxsize; k++) {
						gRows[i][k] = gRows[i][k] ^ gBasis[first][k];
					}

					first = findfirst(i);
				}
			}
			if (first != -1) {
				for (j = 0; j < maxsize; j++) {
					gBasis[first][j] = gRows[i][j];
					answers[i][j] = gRows[i][j];  //�������̵���Ԫ����������firstToRow
				}
				ifBasis[first] = 1;
			}

		}

	}
	//cout << rank << " done process!!!!" << endl << endl;
	for (i = rank + 1; i < total; i++) {
		for (j = begin; j < end; j++) {

			MPI_Send(&answers[j][0], maxsize, MPI_INT, i, 2, MPI_COMM_WORLD);//2�Ǹý��̵���Ԫ�����������Ϊ֮����̵���Ԫ��
		}
	}

	if (rank == total - 1) {
		end_time = MPI_Wtime();
		cout << "MPI�Ż��汾��ʱ�� " << 1000 * (end_time - start_time) << "ms" << endl;
		writeResult_MPI(out_mpi);
		out_mpi.close();
	}
	MPI_Finalize();

}

void GE_MPI_AVX_omp(int argc, char* argv[]) {
	int flag;
	double start_time = 0;
	double end_time = 0;
	MPI_Init(&argc, &argv);
	int total = 0;
	int rank = 0;
	int i = 0;
	int j = 0;
	int begin = 0, end = 0;
	MPI_Status status;
	MPI_Comm_size(MPI_COMM_WORLD, &total);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (rank == 0) {
		flag = readRowsFrom(0);     //��ȡ����Ԫ��
		num = (flag == -1) ? maxrow : flag;
		begin = rank * num / total;
		end = (rank == total - 1) ? num : (rank + 1) * (num / total);
		for (i = 1; i < total; i++) {
			MPI_Send(&num, 1, MPI_INT, i, 0, MPI_COMM_WORLD);//0�Ǳ���Ԫ������
			int b = i * (num / total);
			int e = (i == total - 1) ? num : (i + 1) * (num / total);
			for (j = b; j < e; j++) {
				MPI_Send(&gRows[j][0], maxsize, MPI_INT, i, 1, MPI_COMM_WORLD);//1ʱ����Ԫ������
			}
		}

	}
	else {
		//cout << rank << " recving" << endl;
		MPI_Recv(&num, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
		//cout << rank << " num:" << num << endl;
		begin = rank * (num / total);
		end = (rank == total - 1) ? num : (rank + 1) * (num / total);
		for (i = begin; i < end; i++) {
			MPI_Recv(&gRows[i][0], maxsize, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
		}
	}

	//cout << rank << "  " << begin << "   " << end << endl << endl;

	MPI_Barrier(MPI_COMM_WORLD);  
	start_time = MPI_Wtime();
#pragma omp parallel  num_threads(NUM_THREADS),private(i,j)
#pragma omp for ordered schedule(guided)
	for (i = begin; i < end; i++) {
		//cout << omp_get_thread_num() << " from " << rank << " processing " << i << endl;
		int first = findfirst(i);
		while (first != -1) {    

			//int first = findfirst(i);      

			if (ifBasis[first] == 1) {  
				for (j = 0; j + 8 < maxsize; j += 8) {
					__m256i vij = _mm256_loadu_si256((__m256i*) & gRows[i][j]);
					__m256i vj = _mm256_loadu_si256((__m256i*) & gBasis[first][j]);
					__m256i vx = _mm256_xor_si256(vij, vj);
					_mm256_storeu_si256((__m256i*) & gRows[i][j], vx);
				}
				for (; j < maxsize; j++) {
					gRows[i][j] = gRows[i][j] ^ gBasis[first][j];
				}
				first = findfirst(i);
			}
			else {   
				//tmpAns.push_back(first);
				//ifDone[i] = 1;
#pragma omp ordered
				if (rank == 0) {
					//cout << omp_get_thread_num() << " from " << rank << " writing " << i << endl;
					while (ifBasis[first] == 1) {
						for (j = 0; j + 8 < maxsize; j += 8) {
							__m256i vij = _mm256_loadu_si256((__m256i*) & gRows[i][j]);
							__m256i vj = _mm256_loadu_si256((__m256i*) & gBasis[first][j]);
							__m256i vx = _mm256_xor_si256(vij, vj);
							_mm256_storeu_si256((__m256i*) & gRows[i][j], vx);
						}
						for (; j < maxsize; j++) {
							gRows[i][j] = gRows[i][j] ^ gBasis[first][j];
						}
						first = findfirst(i);
					}
					if (first != -1) {
						for (j = 0; j + 8 < maxsize; j += 8) {
							__m256i vij = _mm256_loadu_si256((__m256i*) & gRows[i][j]);
							_mm256_storeu_si256((__m256i*) & gBasis[first][j], vij);
							_mm256_storeu_si256((__m256i*) & answers[i][j], vij);
						}
						for (; j < maxsize; j++) {
							gBasis[first][j] = gRows[i][j];
							answers[i][j] = gRows[i][j];
						}
						ifBasis[first] = 1;  
					}
				}
				first = -1;
				//ifBasis[first] = 1;
				//break;
			}
		}
	}
	//cout << rank << " done for own" << endl;
	for (i = 0; i < rank; i++) {

		int b = i * (num / total);
		int e = b + num / total;
		//cout << rank << " wating " << i << endl;
		for (j = b; j < e; j++) {
			MPI_Recv(&answers[j][0], maxsize, MPI_INT, i, 2, MPI_COMM_WORLD, &status);
			int first = _findfirst(j);
			firstToRow.insert(pair<int, int>(first, j));
		}
#pragma omp for schedule(guided)
		for (j = begin; j < end; j++) {  
			//cout << rank << " doing " << j << endl;

			int first = findfirst(j);
			while ((firstToRow.find(first) != firstToRow.end() || ifBasis[first] == 1) && first != -1) {  
				if (firstToRow.find(first) != firstToRow.end()) {  
					int row = firstToRow.find(first)->second;
					int k = 0;
					for (k = 0; k + 8 < maxsize; k += 8) {
						__m256i vij = _mm256_loadu_si256((__m256i*) & gRows[j][k]);
						__m256i vj = _mm256_loadu_si256((__m256i*) & answers[row][k]);
						__m256i vx = _mm256_xor_si256(vij, vj);
						_mm256_storeu_si256((__m256i*) & gRows[j][k], vx);
					}
					for (; k < maxsize; k++) {
						gRows[j][k] = gRows[j][k] ^ answers[row][k];
					}
					first = findfirst(i);
					//cout << "done 1 :" << first << endl;
				}
				if (ifBasis[first] == 1) {
					int k = 0;
					for (k = 0; k + 8 < maxsize; k += 8) {
						__m256i vij = _mm256_loadu_si256((__m256i*) & gRows[j][k]);
						__m256i vj = _mm256_loadu_si256((__m256i*) & gBasis[first][k]);
						__m256i vx = _mm256_xor_si256(vij, vj);
						_mm256_storeu_si256((__m256i*) & gRows[j][k], vx);
					}
					for (; k < maxsize; k++) {
						gRows[j][k] = gRows[j][k] ^ gBasis[first][k];
					}
					first = findfirst(i);
				}
			}

		}
	}

	if (rank != 0) {
		for (i = begin; i < end; i++) {
			int first = findfirst(i);
			//cout << rank << " doing " << first << endl;		

			if (first != -1) {
				while ((firstToRow.find(first) != firstToRow.end() || ifBasis[first] == 1) && first != -1) { 
					if (firstToRow.find(first) != firstToRow.end()) { 
						int row = firstToRow.find(first)->second;
						int k = 0;
						for (k = 0; k + 8 < maxsize; k += 8) {
							__m256i vij = _mm256_loadu_si256((__m256i*) & gRows[i][k]);
							__m256i vj = _mm256_loadu_si256((__m256i*) & answers[row][k]);
							__m256i vx = _mm256_xor_si256(vij, vj);
							_mm256_storeu_si256((__m256i*) & gRows[i][k], vx);
						}
						for (; k < maxsize; k++) {
							gRows[i][k] = gRows[i][k] ^ answers[row][k];
						}
						first = findfirst(i);
						//cout << "done 1 :" << first << endl;
					}
					if (ifBasis[first] == 1) {
						int k = 0;
						for (k = 0; k + 8 < maxsize; k += 8) {
							__m256i vij = _mm256_loadu_si256((__m256i*) & gRows[i][k]);
							__m256i vj = _mm256_loadu_si256((__m256i*) & gBasis[first][k]);
							__m256i vx = _mm256_xor_si256(vij, vj);
							_mm256_storeu_si256((__m256i*) & gRows[i][k], vx);
						}
						for (; k < maxsize; k++) {
							gRows[i][k] = gRows[i][k] ^ gBasis[first][k];
						}

						first = findfirst(i);
					}
				}
				if (first == -1) {
					continue;
				}
				for (j = 0; j < maxsize; j++) {
					gBasis[first][j] = gRows[i][j];
					answers[i][j] = gRows[i][j]; 
				}
				ifBasis[first] = 1;
			}

		}

	}
	//cout << rank << " done process!!!!" << endl << endl;
	for (i = rank + 1; i < total; i++) {
		for (j = begin; j < end; j++) {

			MPI_Send(&answers[j][0], maxsize, MPI_INT, i, 2, MPI_COMM_WORLD);
		}
	}

	if (rank == total - 1) {
		end_time = MPI_Wtime();
		cout << "MPI+omp+AVX�Ż��汾��ʱ�� " << 1000 * (end_time - start_time) << "ms" << endl;
		writeResult_MPI(out_mpi);
		out_mpi.close();
	}
	MPI_Finalize();

}

int main(int argc, char* argv[]) {

	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

	/*readBasis();
	GE();*/

	//reset();

	/*readBasis();
	GE_MPI(argc, argv);*/

	//reset();

	readBasis();
	GE_MPI_AVX_omp(argc, argv);
}

//8_23045_18748_14325
//5_2362_1226_453
// 
//4_1011_539_263
//3_562_170_53
//2_254_106_53