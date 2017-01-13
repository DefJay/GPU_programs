#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>

#include "C:/Users/educ/Desktop/GPU_programs/timer_test/high_performance_timer/High_performance_timer.h"

using namespace cv;
using namespace std;

void Threshold(int threashold, int width, int height, unsigned char * data);

//global threshold kernel for cuda
//__global__ void threshold_kernel()

//global copy for cuda
__global__ void copy_image(const unsigned char* in, unsigned char* out) {
	//blockIdx == x, threadIdx == y, blockDim == width
	int index = (blockIdx.x + threadIdx.x * blockDim.x) * 4;

	//copy the color channels
	out[index] = in[index];
	out[index + 1] = in[index + 1];
	out[index + 2] = in[index + 2];
	out[index + 3] = in[index + 3];

}

int main(int argc, char** argv) {
	if (argc != 2) {
		cout << "Usage: display_image ImageToLoadAndDisplay" << endl;
	}

	Mat image;
	image = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	if (!image.data) {
		cout << "Could not open or find the image" << endl;
		return -1;
	}

	//convert the color into grey
	cvtColor(image, image, cv::COLOR_RGB2GRAY);


	Threshold(100, image.cols, image.rows, image.data);

	namedWindow("Display window", WINDOW_NORMAL);
	imshow("Display window", image);

	//waitKey(0);


	//=============cuda code================
	//find the size of the image to use later
	int size = image.cols * image.rows * 4;

	cudaError cuda_status;

	unsigned char* in = nullptr;
	unsigned char* out = nullptr;


	try {
		//allocate the GPU buffers for the image
		cuda_status = cudaMalloc((void**)&in, size * sizeof(unsigned char));
		if (cuda_status != cudaSuccess) {
			throw("cudaMalloc of in image failed!");
		}
		cuda_status = cudaMalloc((void**)&out, size * sizeof(unsigned char));
		if (cuda_status != cudaSuccess) {
			throw("cudaMalloc of out image failed!");
		}

		//copy over the image from hos memory to gpu
		cuda_status = cudaMemcpy(in, image.datastart, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
		if (cuda_status != cudaSuccess) {
			throw("cudaMemcpy failed!");
		}
	}
	catch (char * err_message) {
		cout << err_message << endl;
		goto Error;
	}
	
	//call the copy kernel on the GPU with one thread fro each element
	copy_image<<<image.cols, image.rows >>> (in, out);






Error:
	cudaFree(in);
	cudaFree(out);
	return 0;
}



void Threshold(int threshold, int width, int height, unsigned char * data) {
	for (int i = 0; i < (height * width); i++) {
		if (data[i] > threshold) {
			data[i] = 255;
		}
		else {
			data[i] = 0;
		}
	}
}
