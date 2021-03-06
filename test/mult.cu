// NOTE: CUSP's own IO library is so poorly written and has really really no efficiency.
// Compile with  nvcc -I./ -arch=sm_20 -O2 test/mult.cu
#include <cusp/multiply.h>
#include <cusp/io/matrix_market.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/print.h>
#include <vector>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>
const int N = 61578044;//61578171;                                                                              
const int M =345389469; //345439900;
const float DAMPINGFACTOR = 0.85;
const char mtxBinRowFile[] = "/media/tmp/graphchi/data/test4row";
const char mtxBinColFile[] = "/media/tmp/graphchi/data/test4col";
const char mtxBinValFile[] = "/media/tmp/graphchi/data/test4val";
cusp::coo_matrix<int, float, cusp::host_memory> B(N,N,M);
const int niter = 4;
const char mtxFile[] = "/media/tmp/graphchi/data/test4";

void FIXLINE(char *s){
	int l = (int)strlen(s)-1;
	if(s[l] == '\n')s[l]=0;
}

void readBinMatrix(int *row, int *col, float *val, int m){
	FILE *fprow = fopen(mtxBinRowFile,"rb");
	FILE *fpcol = fopen(mtxBinColFile,"rb");
	FILE *fpval = fopen(mtxBinValFile,"rb");
	fread(row, sizeof(int), m, fprow);
	fread(col, sizeof(int), m, fpcol);
	fread(val, sizeof(float), m, fpval);
	fclose(fprow);
	fclose(fpcol);
	fclose(fpval);
}


void my_read_matrix(){
	int cnt=0;
	FILE *fp = fopen(mtxFile,"r");
	char s[1024];
	double diff;
	time_t time0,time1;
	time(&time0);
	while(fgets(s, 1024, fp) != NULL){
		FIXLINE(s);
		char del[] = "\t ";
		if(s[0]=='#' || s[0] == '%') continue;
		char *t;
		int a,b;
		float c;
		t=strtok(s,del);
		a=atoi(t);
		t=strtok(NULL,del);
		b=atoi(t);
		t=strtok(NULL,del);
		c=atof(t);
		B.row_indices[cnt] = a;
		B.column_indices[cnt] = b;
		B.values[cnt] = c;
		cnt++;
	}
	printf("\n");
	time(&time1);
	diff = difftime(time1, time0);
	printf("Reading %d lines takes %.3f\n", cnt, diff);
}
void reportTime(clock_t t0){
	printf("----ELAPSED TIME: %.8fs\n", ((double)clock() - t0)/CLOCKS_PER_SEC);
}

int main(void)
{
	time_t t0,t1;
	double diff;
	clock_t tt0=clock();
	cudaSetDevice(0);

	my_read_matrix();
	reportTime(tt0);

	cusp::coo_matrix<int, float, cusp::device_memory> A(B);
	cusp::array1d<float, cusp::device_memory> x(A.num_cols, 1);
	cusp::array1d<float, cusp::device_memory> y(A.num_rows, 1);
	cusp::array1d<float, cusp::device_memory> z(A.num_rows, (1-DAMPINGFACTOR)/N);

	// for each iteration,  y<-A*x, x<-z, x<-D*y + x
	for(int i=0;i<niter;i++){
		clock_t tc, ti;

		ti = tc = clock();
		cusp::multiply(A, x, y);
		diff = ((float)clock() - tc)/CLOCKS_PER_SEC;
		printf("multiplication takes %.5f\n",diff);

		tc = clock();
		x=z;
		diff = ((float)clock() - tc)/CLOCKS_PER_SEC;
		printf("assignment takes %.5f\n",diff);

		tc = clock();
		cusp::blas::axpy(y,x,DAMPINGFACTOR);
		diff = ((float)clock() - tc)/CLOCKS_PER_SEC;
		printf("blas::axpy takes %.5f\n",diff);

		diff = ((float)clock() - ti)/CLOCKS_PER_SEC;
		printf("iteration takes %.5f\n",diff);
	}
	reportTime(tt0);
	time(&t0);
	cusp::array1d<float, cusp::host_memory> yy(x);
	time(&t1);
	diff = difftime(t1,t0);
	printf("copying host takes %.3f\n",diff);
	reportTime(tt0);
	time(&t0);

	std::vector<float> yyy(yy.begin(), yy.end());
	for(int i=0;i<30;i++)
		printf("%.8f\n",yyy[i]);
	reportTime(tt0);
	return 0;
}

