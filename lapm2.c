// MPI

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>
#include <time.h>

#define S 150
#define maxitr 100000
#define eps 1e-4


int main(int argc, char** argv)
{
    int rank, world_size;
    clock_t start, end;
    start = clock();

    MPI_Init(&argc, &argv); // 初始化MPI环境 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // 获取当前进程的Rank
    MPI_Comm_size(MPI_COMM_WORLD, &world_size); // 获取总进程数

    int Slocal, Sglobal;
    Slocal =(int)ceil((double)(S-2) / world_size) + 2;
    Sglobal = (Slocal-2) * world_size + 2;

    // double* ug = (double*)malloc( S*Sglobal*sizeof(double) ); // 实际计算域S*S
    double* u = (double*)malloc( S*Slocal*sizeof(double) );
    double* u_new = (double*)malloc( S*Slocal*sizeof(double) );
    double* tmp;
    double eglobal, elocal, e, t;
    int i, j, itr;

    if (rank==0) {
        for(j=0; j<S; j++)
            u[j] = 100;
        for(i=1; i<Slocal; i++){
            for(j=0; j<S; j++)
                u[i*S+j] = 0;
        }
    }
    else {
        for(i=0; i<Slocal; i++){
            for(j=0; j<S; j++)
                u[i*S+j] = 0;
        }
    }
    memcpy(u_new, u, S*Slocal*sizeof(double));

    int Slocal2;
    if (rank == (world_size-1)) {
        Slocal2 = S - (Slocal-2)*rank; // 最后一个线程的计算域
    }
    else {
        Slocal2 = Slocal;
    }

    
    MPI_Request rsend1, rrecv1, rsend2, rrecv2;
    int isfinish = 0;
    for(itr=0; itr<maxitr; itr++)
    {
        // 线程间传递重叠区域
        if (rank == 0) {
            MPI_Isend(u+S*(Slocal-2), S, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &rsend1);
            MPI_Irecv(u+S*(Slocal-1), S, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &rrecv1);
            MPI_Wait(&rsend1, MPI_STATUS_IGNORE);
            MPI_Wait(&rrecv1, MPI_STATUS_IGNORE);
        }
        else if (rank == (world_size-1)) {
            MPI_Isend(u+S, S, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &rsend2);
            MPI_Irecv(u, S, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &rrecv2);
            MPI_Wait(&rsend2, MPI_STATUS_IGNORE);
            MPI_Wait(&rrecv2, MPI_STATUS_IGNORE);
        }
        else {
            MPI_Isend(u+S*(Slocal-2), S, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &rsend1);
            MPI_Irecv(u+S*(Slocal-1), S, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &rrecv1);
            MPI_Isend(u+S, S, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &rsend2);
            MPI_Irecv(u, S, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &rrecv2);
            MPI_Wait(&rsend1, MPI_STATUS_IGNORE);
            MPI_Wait(&rsend2, MPI_STATUS_IGNORE);
            MPI_Wait(&rrecv1, MPI_STATUS_IGNORE);
            MPI_Wait(&rrecv2, MPI_STATUS_IGNORE);
        }
        // 迭代计算
        elocal = 0;
        for (i=1; i<Slocal2-1; i++)
        {
            for (j=1; j<S-1; j++)
            {
                u_new[i*S+j] = (u[(i-1)*S+j]+u[(i+1)*S+j]+u[i*S+j+1]+u[i*S+j-1])/4;
                e = fabs(u[i*S+j]-u_new[i*S+j]);
                if (e>elocal)
                    elocal = e;
            }
        }
        // 交换指针
        tmp = u_new;
        u_new = u;
        u = tmp;
        // printf("Iteration Number %d\n",itr);
        // oput(u);
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Scan(&elocal, &eglobal, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Bcast(&eglobal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        if(eglobal<=eps)
        {
            break;
        }
    }
    /*MPI_Gather(
        u+S, S*(Slocal-2), MPI_DOUBLE,
        ug+S, S*(Slocal-2), MPI_DOUBLE,
        0, MPI_COMM_WORLD);*/
    
    if (rank == 0) {
        end = clock();
        printf("S = %d\n", S);
        printf("end at %d itr\n", itr);
        printf("time = %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
    }

    // 以写入模式打开文件
    MPI_File fh;
    MPI_Status status;
    MPI_File_open(MPI_COMM_WORLD, "mpioutput2.bin", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

    // 每个进程计算写入文件的偏移量
    MPI_Offset offset = rank * S * (Slocal-2) * sizeof(double);

    // 设置文件指针到正确位置
    MPI_File_seek(fh, offset, MPI_SEEK_SET);

    // 写入文件
    if (rank == (world_size-1)) {
        MPI_File_write(fh, u, S*Slocal2, MPI_DOUBLE, &status);
    } else {
        MPI_File_write(fh, u, S*(Slocal-2), MPI_DOUBLE, &status);
    }

    // 关闭文件
    MPI_File_close(&fh);

    free(u); free(u_new);
    MPI_Finalize();
    return 0;
}