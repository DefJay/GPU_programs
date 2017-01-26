#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../high_performance_timer/High_performance_timer.h" 
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <omp.h>

using namespace std;

HighPrecisionTime htp;

#define USE_OMP

#if defined(_DEBUG)
#define GIGA	(1 << 20)
#else
#define GIGA	(1 << 30)
#endif

#define BMSIZE (GIGA / 8)
#define MAX_PATTERN_LENGTH 256

__constant__ char dev_pattern[MAX_PATTERN_LENGTH];
__constant__ int dev_pattern_size;
__device__ char * dev_buffer = nullptr;
__device__ unsigned char * dev_bitmap = nullptr;

int search_cpu(char * buffer, int buffer_size, char * pattern, int pattern_size, unsigned char * bitmap, int bitmap_size) {
	int retval = 0;

	#if defined(USE_OMP)
	#pragma omp parallel for
	#endif
	for (int c_index = 0; c_index < buffer_size; c_index++) {
		//index for marching through the pattern
		int p_index;

		for (p_index = 0; p_index < pattern_size; p_index++) {
			if (tolower(*(buffer + c_index + p_index)) != *(pattern + p_index)) {
				break;
			}
		}

		if (p_index == pattern_size) {
			int byte_num = c_index % 8;

			if (byte_num < bitmap_size) {
				int bit_num = c_index % 8;
				#if defined(USE_OMP)
				#pragma omp critical
				#endif
				{
					*(bitmap + byte_num) |= (1 << bit_num);
					retval++;
				}
			}

		}
	}
	return retval;
}


/*	CStringToLower() - this function flattens a c string to all
lower case. It marches through memory until a null byte is
found. As such, some may consider this function unsafe.

By flattening the pattern, we can eliminate a tolower in
the search function - a potentially big win.

The original pointer is returned so that the function can be
used in an assignment statement.
*/
char * Cstring_to_lower(char * s) {
	char * retval = s;

	for (; *s != NULL; s++) {
		*s = tolower(*s);
	}
	return retval;
}

//checks to see if cuda returned success and if not throws the error message
inline void check_cuda_throw(cudaError_t t, const string & message) {
	if (t != cudaSuccess) {
		throw message;
	}
}

int main(int argc, char * argv) {
	
	char * host_buffer = nullptr;
	unsigned char * host_bitmap = nullptr;
	unsigned char * check_bitmap = nullptr;

	//read in the file
	ifstream file("C:/Users/educ/Documents/enwiki-latest-abstract.xml"); 

	#if defined(USE_OMP)
		cout << "OMP enabled on " << omp_get_max_threads() << " threads." << endl;
	#endif
	

	try {
		//check to make sure they entered in a word to search using
		//the second command argument

		cout << argv[1] << endl;
		if (argc < 2) {
			throw string("First argument must be the target string.");
		}

		char * pattern = Cstring_to_lower(&argv[1]);
		int pattern_size = strlen(pattern);

	
		//make sure the file was actually opened
		if (!file.is_open()) {
			throw string("ERROR READING FILE");
		}


		//allocate the space on the heap for the buffer and bitmaps
		host_buffer = new char[GIGA];
		host_bitmap = new unsigned char[BMSIZE]();
		check_bitmap = new unsigned char[BMSIZE]();



		htp.TimeSinceLastCall();
		//read in the file into the buffer
		file.read(host_buffer, GIGA);

		//make sure that the file reads into the buffer
		if (!file) {
			throw string("ERROR READING THE CHARACTERS INTO THE BUFFER");
		}
		double read_time = htp.TimeSinceLastCall();

		cout << GIGA << " bytes read from disk in " << read_time << " seconds at a speed of " << GIGA / read_time / double(1 << 30) << " GB / sec." << endl;

		//set up the data for the cuda part of the project
		check_cuda_throw(cudaSetDevice(0), string("cudaSetDevice(0) failed on line ") + to_string(__LINE__));
		check_cuda_throw(cudaMalloc(&dev_buffer, GIGA), string("cudaMalloc failed on line ") + to_string(__LINE__));
		check_cuda_throw(cudaMalloc(&dev_bitmap, BMSIZE), string("cudaMalloc failed on line ") + to_string(__LINE__));
		check_cuda_throw(cudaMemset(dev_bitmap, 0, BMSIZE), string("cudaMemset failed on line ") + to_string(__LINE__));
		check_cuda_throw(cudaMemcpyToSymbol(dev_pattern, pattern, pattern_size, 0), string("cudaMemcpyToSymbol failed on line ") + to_string(__LINE__));
		check_cuda_throw(cudaMemcpyToSymbol(dev_pattern_size, &pattern_size, sizeof(int), 0), string("cudaMemcpyToSymbol failed on line ") + to_string(__LINE__));

		//copy the data over to the GPU
		htp.TimeSinceLastCall();
		check_cuda_throw(cudaMemcpy(dev_buffer, host_buffer, GIGA, cudaMemcpyHostToDevice), string("cudaMemcpy failed on line ") + to_string(__LINE__));
		double copy_time = htp.TimeSinceLastCall();
		cout << GIGA << " data bytes copied to GPU in " << copy_time << " seconds at " << GIGA / copy_time / double(1 << 30) << " GB / second." << endl;

		//search through the document using the CPU
		htp.TimeSinceLastCall();
		int matches_found = search_cpu(host_buffer, GIGA, pattern, pattern_size, host_bitmap, BMSIZE);
		double cpu_search_time = htp.TimeSinceLastCall();

		cout << "search_cpu found the word " << pattern << " " << matches_found << " times in " << cpu_search_time << " seconds.";
		
		int threads_per_block = 1024;
		dim3 grid(1024, 1024);

	}

	catch (string error){
		cout << error << endl;
	}




	//clean and clear out the buffer
	if (host_buffer != nullptr) {
		delete[] host_buffer;
	}

	if (host_bitmap != nullptr) {
		delete[] host_bitmap;
	}

	if (check_bitmap != nullptr) {
		delete[] check_bitmap;
	}



	//close the file
	if (file.is_open()) {
		file.close();
	}

#if defined(WIN64) || defined(WIN32)
	cout << endl;
	system("pause");
#endif

	return 0;
}