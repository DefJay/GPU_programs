#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../high_performance_timer/High_performance_timer.h" 
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>

using namespace std;

HighPrecisionTime htp;

#define GIGA (1 << 30)
#define BMSIZE (GIGA / 8)


int main(int argc, char * argv) {

	//malloc room for a 1 GB array
	char * buffer = new char [GIGA]();

	//malloc the bit map
	char * bit_map = new char[BMSIZE]();

	//read in the file
	ifstream file("C:/Users/educ/Documents/enwiki-latest-abstract.xml"); 


	

	try {
		//make sure the file was actually opened
		if (!file.is_open()) {
			throw string("ERROR READING FILE");
		}

		htp.TimeSinceLastCall();
		//read in the file into the buffer
		file.read(buffer, GIGA);

		cout << "reading in the file took: " << htp.TimeSinceLastCall() << "seconds" << endl;

		//make sure that the file reads into the buffer
		if (!file) {
			throw string("ERROR READING THE CHARACTERS INTO THE BUFFER");
		}

	}

	catch (string error){
		cout << error << endl;
	}




	//clean and clear out the buffer
	if (buffer != nullptr) {
		delete[] buffer;
	}

	if (bit_map != nullptr) {
		delete[] bit_map;
	}

	//close the file
	if (file.is_open()) {
		file.close();
	}

	return 0;
}