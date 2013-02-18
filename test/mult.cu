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
const int N = 61578170;
const int M = 345439900;
const float DAMPINGFACTOR = 0.85;
cusp::coo_matrix<int, float, cusp::host_memory> B(N,N,M);
const int niter = 4;
const char mtxFile[] = "/media/tmp/graphchi/data/test3";

void FIXLINE(char *s){
	int l = (int)strlen(s)-1;
	if(s[l] == '\n')s[l]=0;
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

int main(void)
{
	time_t t0,t1;
	double diff;
	cudaSetDevice(0);

	my_read_matrix();

	cusp::coo_matrix<int, float, cusp::device_memory> A(B);
	cusp::array1d<float, cusp::device_memory> x(A.num_cols, 1);
	cusp::array1d<float, cusp::device_memory> y(A.num_rows, 1);
	cusp::array1d<float, cusp::device_memory> z(A.num_rows, (1-DAMPINGFACTOR)/N);

	// for each iteration,  y<-A*x, x<-z, x<-D*y + x
	for(int i=0;i<niter;i++){
		clock_t tc;
		tc = clock();
		cusp::multiply(A, x, y);
		x=z;
		cusp::blas::axpy(y,x,DAMPINGFACTOR);
		diff = ((float)clock() - tc)/CLOCKS_PER_SEC;
		printf("iteration takes %.5f\n",diff);
	}
	time(&t0);
	cusp::array1d<float, cusp::host_memory> yy(x);
	time(&t1);
	diff = difftime(t1,t0);
	printf("copying host takes %.3f\n",diff);
	time(&t0);

	std::vector<float> yyy(yy.begin(), yy.end());
	for(int i=0;i<30;i++)
		printf("%.8f\n",yyy[i]);
	return 0;
}

