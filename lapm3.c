// OpenMP

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define S 200
#define maxitr 100000
#define eps 1e-4


void entrow(int i, double* u)
{
    int j;
    printf("Enter the value of u[%d,j],j=0~%d\n",i,S);
    for(j=0;j<S;j++)
        scanf("%lf",&u[i*S+j]);
}

void entcol(int j, double* u)
{
    int i;
    printf("Enter the value of u[i,%d],""i=1~%d\n",j,S-1);
    for(i=1;i<S-1;i++)
        scanf("%lf",&u[i*S+j]);
}

void oput(double* u)
{
    FILE *fp = fopen("ompoutput.bin", "wb");
    if (fp == NULL) {
        perror("Error opening file");
        return;
    }
    fwrite(u, sizeof(double), S*S, fp);
    fclose(fp);
}

int main()
{
    int nthreads;
    printf("Input OpenMP parallel threads:\n");
    scanf("%d", &nthreads);
    omp_set_num_threads(nthreads);
    int block_size = (int)ceil((double)(S-2) / nthreads);
    
    double start2 = omp_get_wtime();
    
    double* u = (double*)malloc( S*S*sizeof(double) );
    double* u_new = (double*)malloc( S*S*sizeof(double) );
    double* tmp;
    double elocal = 100;
    int itr;
    for(int j=0; j<S; j++) {
        u[j] = 100;
    }
    for(int i=1; i<S; i++){
        for(int j=0; j<S; j++) {
            u[i*S+j] = 0;
        }
    }
    memcpy(u_new, u, S*S*sizeof(double));
    // printf("Enter the Boundary Condition\n");
    // entrow(0,u); entrow(S,u);
    // entcol(0,u); entcol(S,u);
    for(itr=0; itr<maxitr && elocal>eps; itr++)
    {
        elocal = 0;
        #pragma omp parallel shared(u, u_new) reduction(max:elocal)
        {
            int k = omp_get_thread_num();
            int jmin = block_size * k + 1;
            int jmax = block_size * (k + 1);
            if (jmax > S - 2)
                jmax = S - 2;
            for (int i=1; i<S-1; i++) {
                for (int j=jmin; j<=jmax; j++) {
                    u_new[i*S+j] = (u[(i-1)*S+j]+u[(i+1)*S+j]+u[i*S+j+1]+u[i*S+j-1])/4;
                    double e = fabs(u[i*S+j]-u_new[i*S+j]);
                    if (e > elocal)
                        elocal = e;
                }
            }
        }
        // 交换指针
        tmp = u_new;
        u_new = u;
        u = tmp;
    }
    printf("end at %d itr\n", itr);
    double end2 = omp_get_wtime();
    printf("parallel time: %.6f s\n", end2 - start2);
    oput(u);

    free(u); free(u_new);
    return 0;
}