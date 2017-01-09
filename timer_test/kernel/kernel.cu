#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../high_performance_timer/High_performance_timer.h" 

#include <iostream>
#include <string>
#include <stdlib.h>
#include <time.h>
#include<omp.h>


using namespace std;

bool mem_alloc(int **a, int **b, int **c, int size);
void clean_up(int *a, int *b, int *c);
void fill_arrays(int *a, int *b, int *c, int size);
void add_vec_serial_CPU(int * a, int * b, int * c, int size);

cudaError_t add_array_cuda(int *c, const int *a, const int, *b, unsigned int size);



int main(int argc, char * argv[]) {
	//seed random with time
	srand((unsigned)time(NULL));

	//set up the timer
	HighPrecisionTime htp;

	//set the second argument to be 1000 by default
	int size = 1000;
	//set the iterations to be 100 by default
	int iter = 100;

	//declare the variables
	int *a = nullptr;
	int *b = nullptr;
	int *c = nullptr;

	//check to see the user added a second argument
	//if they did, then change the size to that new argument
	if (argc > 1) {
		size = stoi(argv[1]);
	}
	cout << "the size of the array is: " << size << endl;

	//check to see if the user added a third argument
	//if they did, then change the size to that new argument
	if (argc > 2) {
		iter = stoi(argv[2]);
	}
	cout << "the number of iterations is: " << iter << endl;

	//try to allocate the memory
	try {
		if (!mem_alloc(&a, &b, &c, size)) {
			throw("did not correctly allocate!");
		}
		cout << "memory has been allocated!" << endl;
	}
	//if it doesn't work print out the error message and continue
	//to clean up
	catch(char * err_message) {
		cout << err_message << endl;
	}

	fill_arrays(a, b, c, size);

	double t = 0;


	

	htp.TimeSinceLastCall();
	for (int i = 0; i < iter; i++) {
		add_vec_serial_CPU(a, b, c, size);
		t = t + htp.TimeSinceLastCall();
	}
		
	t = t / iter;
	cout << "Add vec serial CPU took:   " << t << "  seconds!" << endl;





	
	clean_up(a, b, c);
	return 0;
}



//---------------------------------------------------------------------------
//function to allocate memory
bool mem_alloc(int **a, int **b, int **c, int size) {
	//set up the return value to be false
	bool retval = false;

	//allocate memory for all the arrays and size
	*a = (int *)malloc(sizeof(int) * size);
	*b = (int *)malloc(sizeof(int) * size);
	*c = (int *)malloc(sizeof(int) * size);

	//check to make sure they properly allocated
	//if they were then change retval to true
	if (*a != NULL || *b != NULL || *c != NULL) {
		retval = true;
	}

	return retval;
}

//---------------------------------------------------------------------------
//function for cleaning up and freeing the data
void clean_up(int *a, int *b, int *c) {
	free(a);
	free(b);
	free(c);

	if (a != nullptr) {
		a = nullptr;
	}
	if (a != nullptr) {
		b = nullptr;
	}
	if (a != nullptr) {
		c = nullptr;
	}
}

//---------------------------------------------------------------------------
void fill_arrays(int *a, int *b, int *c, int size) {
	//fill in the arrays 
	for (int i = 0; i < size; i++) {
		a[i] = rand() % 20 + 1;
		b[i] = rand() % 20 + 1;
		c[i] = 0;
	}
}




//---------------------------------------------------------------------------
void add_vec_serial_CPU(int * a, int * b, int * c, int size) {
	//add a and b and save it into c
	for (int i = 0; i < size; i++) {
		c[i] = a[i] + b[i];
	}
}
