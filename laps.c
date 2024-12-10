// serial ver

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define S 150
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
    FILE *fp = fopen("serialoutput.bin", "wb");
    if (fp == NULL) {
        perror("Error opening file");
        return;
    }
    fwrite(u, sizeof(double), S*S, fp);
    fclose(fp);
}

int main()
{
    clock_t start, end;
    start = clock();
    double* u = (double*)malloc( S*S*sizeof(double) );
    double* u_new = (double*)malloc( S*S*sizeof(double) );
    double* tmp;
    double mer, e, t;
    int i, j, itr;
    for(j=0; j<S; j++) {
        u[j] = 100;
    }
    for(i=1; i<S; i++){
        for(j=0; j<S; j++) {
            u[i*S+j] = 0;
        }
    }
    memcpy(u_new, u, S*S*sizeof(double));
    // printf("Enter the Boundary Condition\n");
    // entrow(0,u); entrow(S,u);
    // entcol(0,u); entcol(S,u);
    for(itr=0; itr<maxitr; itr++)
    {
        mer = 0;
        for(i=1; i<S-1; i++)
        {
            for(j=1; j<S-1; j++)
            {
                u_new[i*S+j] = (u[(i-1)*S+j]+u[(i+1)*S+j]+u[i*S+j+1]+u[i*S+j-1])/4;
                e = fabs(u[i*S+j] - u_new[i*S+j]);
                if(e>mer)
                    mer = e;
            }
        }
        // 交换指针
        tmp = u_new;
        u_new = u;
        u = tmp;
        // printf("Iteration Number %d\n",itr);
        // oput(u);
        if(mer<=eps)
            break;
    }
    end = clock();
    printf("end at %d itr\n", itr);
    printf("time = %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
    oput(u);

    free(u); free(u_new);
    return 0;
}