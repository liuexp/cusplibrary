#include <cusp/io/matrix_market.h>
#include <cusp/coo_matrix.h>
#include <cusp/print.h>
#include <algorithm>
#include <vector>
#include <set>
#include <utility>
#include <ctime>
const int N = 23026589;
const int M = 324874844; 
const char outFile[] = "test3";
const float probability = 0.25;
cusp::coo_matrix<int,float,cusp::host_memory> A(N,N,M);
int maxV=0,lines=0;
std::vector <std::pair<int,int> > invData;
std::vector <std::pair<int, std::pair<int, float> > > outData;
std::map<int, int> mapped;
int outDegree[61578414];

void FIXLINE(char *s){
	int l = (int)strlen(s)-1;
	if(s[l] == '\n')s[l]=0;
}

void addVertex(int a){
	if(mapped.find(a) == mapped.end()){
		mapped[a] = mapped.size();
		maxV = max(maxV, a);
	}
}

void readConv(float prob){
	time_t time0,time1;
	double diff;
	char s[1024];
	FILE *fp = fopen("twitter_rv.net","r");
	int originalMaxV = 0;
	memset(outDegree, 0, sizeof(outDegree[0])*61578414);
	int curline = 0;
	time(&time0);
	srand(time(NULL));
	while(fgets(s, 1024, fp) != NULL ){
		FIXLINE(s);
		char del[] = "\t ";
		if(s[0]=='#' || s[0] == '%') continue;
		double tmp = rand()/(double)RAND_MAX;
		if(tmp>prob)continue;
		char *t;
		int a,b;
		t=strtok(s,del);
		a=atoi(t);
		t=strtok(NULL,del);
		b=atoi(t);
		originalMaxV = max(originalMaxV, max(a,b));
		invData.push_back(std::make_pair(b,a));
		curline++;
	}
	time(&time1);
	diff = difftime(time1, time0);
	printf("here %d lines reading takes %.3f\n",curline,diff);
	sort(invData.begin(),invData.end());
	int n=invData.size();
	time(&time0);
	for(int i=0;i<n;i++){
		int v=invData[i].first,u=invData[i].second;
		outDegree[u]++;
	}
	time(&time1);
	diff = difftime(time1, time0);
	printf("here counting outdegrees takes %.3f\nstart removing redundant vertices and renaming the rest\n", diff);
	time(&time0);
	for(int i=0;i<n;i++){
		int a = invData[i].second, b = invData[i].first;
		if(outDegree[a] == 0 || outDegree[b] == 0)
			continue;
		lines++;
		addVertex(a);
		addVertex(b);
	}
	time(&time1);
	diff = difftime(time1, time0);
	printf("unique renaming takes %.3f\n",diff);
	printf("%d,%d\n", maxV, lines);
	time(&time0);
	FILE *fout = fopen(outFile,"w");
	for(int i=0;i<n;i++){
		int a = invData[i].second, b = invData[i].first;
		//if(outDegreeTemp[a] == 0||outDegreeTemp[b] ==0)continue;
		if(outDegree[a] == 0 || outDegree[b] == 0)continue;
		outData.push_back(std::make_pair(mapped[b], std::make_pair(mapped[a], 1.0/outDegree[b])));
	}
	sort(outData.begin(), outData.end());
	int m = outData.size();
	for(int i=0;i<m;i++){
		int a = outData[i].first, b = outData[i].second.first;
		float c = outData[i].second.second;
		fprintf(fout,"%d %d %.8f\n", a, b, c);
	}
	fclose(fout);
	fclose(fp);
}

void writeConv(){
	int cnt=0;
	FILE *fp = fopen(outFile,"r");
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
		A.row_indices[cnt] = a;
		A.column_indices[cnt] = b;
		A.values[cnt] = c;
		cnt++;
		printf("\r%d",cnt);
		fflush(stdin);
	}
	printf("\n");
	time(&time1);
	diff = difftime(time1, time0);
	printf("here construction takes %.3f\n", diff);
	time(&time0);
	cusp::io::write_matrix_market_file(A, "A.mtx");
	time(&time1);
	diff = difftime(time1, time0);
	printf("here output takes %.3f\n", diff);
	fclose(fp);
}

int main()
{
	readConv(probability);
	//writeConv();
	return 0;
}

