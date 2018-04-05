// romberg.cpp : main project file.

#include <omp.h>
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include "mpi.h"


inline bool Test(double sequent, double parallel)
{
	bool OK = true;
	if (abs(sequent - parallel) > 0.0001)
		OK = false;

	std::cout << "\n Check the results ...";
	if (OK != true)
		std::cout << "Warning!!! Something went wrong." << std::endl;
	else
		std::cout << "Successfully!!!" << std::endl;

	return OK;
}

inline void CheckResults(double sequentTimeWork, double parallTimeWork)
{
	std::cout << "\n Who is faster? ...";
	if (parallTimeWork < sequentTimeWork)
		std::cout << " Parallel algorithm" << std::endl;
	else
		std::cout << " Sequential algorithm" << std::endl;

	std::cout.precision(3);
	std::cout.setf(std::ios::fixed);
	std::cout << " Speedup: " << sequentTimeWork / parallTimeWork << std::endl;
}

double f(double x, double y) {

	return log(x*x*y - log(x)) / sqrt(y*y + x);
}
double TrapLinear(int n) {
	double h1, h2;
	double a = 1.0, b = 2.0, c = 2.0, d = 3.0;

	h1 = (b - a) / n;
	h2 = (d - c) / n;

	double sum = 0;
	double d1 = 0, d2 = 0;

	for (int i = 0; i <= n; i++)
	{
		if (i == 0 || i == n) d1 = 0.5; else d1 = 1;

		for (int j = 0; j <= n; j++) {
			if (j == 0 || j == n) d2 = 0.5; else d2 = 1;
			sum += d1*d2*f(a + (i * h1), c + (j * h2));
		}
	}
	return h1*h2*sum;
}

double rombergLinear(int MAX)
{


	double *s = new double[MAX];
	double var;

	for (int i = 1; i< MAX; i++)
		s[i] = 1;


	for (int k = 1; k < MAX; k++)
	{
		for (int i = 1; i <= k; i++)
		{
			if (i == 1)
			{
				var = s[i];
				s[i] = TrapLinear(pow(2, k - 1));//     // sub-routine Trap

			}                                       // integrated from 0 and 1
													/* pow() is the number of subdivisions*/
			else
			{
				s[k] = (pow(4, i - 1)*s[i - 1] - var) / (pow(4, i - 1) - 1);

				var = s[i];
				s[i] = s[k];
			}

		}

	}

	return s[MAX - 1];


}
double wtgMPI(int proc_num, int proc_rank, int n)
{

	double a = 1.0, b = 2.0, c = 2.0, d = 3.0; //пределы интегрирования по х(a,b), y(c,d)
	double integral = 0., d1 = 0., d2 = 0.;  //начальное значение интеграла

	double h1 = (b - a) / n;
	double h2 = (d - c) / n;


#pragma omp parallel for shared(a, c, h1, h2, proc_rank, proc_num) private(d1, d2) reduction(+:integral)
	for (int i = proc_rank; i < n + 1; i = i + proc_num)
	{
		d1 = 1;
		if (i == 0 || i == n) d1 = 0.5;
		for (int j = 0; j < n + 1; j++)
		{
			d2 = 1;
			if (j == 0 || j == n) d2 = 0.5;
			integral += d1*d2*f(a + (i * h1), c + (j * h2));
		}
	}
	integral *= h1*h2;

	return integral;
}


double rombergParallelMPI(int proc_num, int proc_rank, int MAX)
{

	double *s = new double[MAX];
	int i, k;
	double var;
	for (int i = 1; i< MAX; i++)
		s[i] = 1;

	for (k = 1; k < MAX; k++)
	{

		for (i = 1; i <= k; i++)
		{

			if (i == 1)
			{
				var = s[i];
				double a = wtgMPI(proc_num, proc_rank, pow(2, k - 1));;
				MPI_Allreduce(&a, &s[i], 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			}
			/* pow() is the number of subdivisions*/
			else
			{
				s[k] = (pow(4, i - 1)*s[i - 1] - var) / (pow(4, i - 1) - 1);

				var = s[i];
				s[i] = s[k];
			}

		}

	}

	return s[MAX - 1];


}


void set_thread_affinity() {
#pragma omp parallel default(shared)
{
	DWORD_PTR mask = (1 << omp_get_thread_num());
	SetThreadAffinityMask(GetCurrentThread(), mask);
}
}
int main(int argc, char* argv[])
{
	omp_set_dynamic(0);      // запретить библиотеке openmp менять число потоков во время исполнения
							 //
	int MAX = 16;
	set_thread_affinity();
	int proc_num, proc_rank;
	double I = 0;
	double Ifinale = 0;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &proc_num);
	MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

	int rootProc;
	int n;
	MPI_Status status;
	double parallel_start = 0., parallel_end = 0.;
	double Sequent_integration = 0., Parallel_integration = 0.;
	double t1, t2, parallTimeWork, sequentTimeWork;

	for (int i = 2; i < 20; i = 2 * i) {
		for (int j = 2; j < 20; j = 2 * j) {
			omp_set_num_threads(j);
			if (proc_rank == 0) { //MASTER
				std::cout << "num_threads : " << j << std::endl;
				std::cout << "N - pow : " << i << std::endl;

				t1 = MPI_Wtime();
				Sequent_integration = rombergLinear(i);
				t2 = MPI_Wtime();
				sequentTimeWork = (t2 - t1) * 1000.0;
				std::cout << "\n ******  rombergLinear() integrate ******\n";
				std::cout << "rombergLinear() integrate = " << Sequent_integration << std::endl;
				std::cout << " time = " << sequentTimeWork << "ms\n ***************************";
			}

			parallel_start = MPI_Wtime();
			I = rombergParallelMPI(proc_num, proc_rank, i);
			parallel_end = MPI_Wtime();

			if (proc_rank == 0) {
				double parallTimeWork = (parallel_end - parallel_start) * 1000.0;
				std::cout << "\n rombergParallel() integrate = " << I;
				if (Test(Sequent_integration, I))
					std::cout << " time = " << parallTimeWork << "ms\n ***************************";
				CheckResults(sequentTimeWork, parallTimeWork);
				std::cout << "\n ***************************\n";

			}
		}
	}
	MPI_Finalize();

	return 0;
}