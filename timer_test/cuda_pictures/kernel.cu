#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include <iomanip>

//#include"../high_performance_timer/High_performance_timer.h"

using namespace cv;
using namespace std;

Mat host_image;
Mat orig_image;
Mat temp_image;
Mat final_image;
unsigned char * device_src = nullptr;
unsigned char * device_dst = nullptr;
//HighPrecisionTime t;
size_t image_bytes;
string window_name("Ouput");
const int THRESHOLD_SLIDER_MAX = 256;
int threshold_slider = 0;
int inner_slider = 0;
int outer_slider = 100;
int blur_slider = 0;
double gpu_accumulator = 0;
int gpu_counter = 0;
double cpu_accumulator = 0;
int cpu_counter = 0;
bool gpu_mode = true;

typedef unsigned char ubyte;

//test kernel for box filtering
ubyte test_kernel[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1 };

int one_deminsional(int x, int y, int width) {
	return (x + y * width);
}


void Threshold(Mat & image, int t)
{
	assert(image.channels() == 1);
	unsigned char lt = static_cast<unsigned char>(t);
	const long long e = reinterpret_cast<long long>(image.data + image.cols * image.rows);

	if (t == 256)
	{
		memset(image.data, 0, image.rows * image.cols);
	}
	else
	{
		for (long long p = reinterpret_cast<long long>(image.data); p < e; p++)
		{
			*((unsigned char *)p) = (*((unsigned char *)p) >= lt) ? 255 : 0;
		}
	}
}

void box_filter(ubyte * source, ubyte * dest, int width, int height, ubyte * kernel, int kw, int kh) {
/*
	//walk through the y position in the image 
	for (int y = 0; y < height; y++) {
		//walks through every x position in the image
		for (int x = 0; x < width; x++) {
			//reset the sum for the pixel we are currently working on to be 0
			int sum = 0;

			//points to the middle element of the row of the kernel
			//used to walk through each row of the kernel
			ubyte * kernel_p = kernel + half_kernel_size;

			//walk through the rows of the kernel
			for (int y_offset = -half_kernel_size; y_offset <= half_kernel_size; y_offset++, kernel_p += kernel_size) {
				//checks for the sides of the image
				if (y_offset + y < 0 || y_offset + y >= height) {
					continue;
				}

				//finds the value of the middle elements of the row
				sum += *(src + (y_offset + y) * width + x) * *kernel_p;

				//goes through the current row and finds the sums for the x values
				//to the side of the middle element of the row that kernel_p points to
				for (int offset_x = 1; offset_x <= half_kernel_size; offset_x++) {
					if (x - offset_x >= 0) {
						sum += *(src + (y_offset + y) * width - offset_x + x) * *(kernel_p - offset_x);
					}
					if (x + offset_x < width) {
						sum += *(src + (y_offset + y) * width + offset_x + x) * *(kernel_p + offset_x);
					}

					//assign the new values to the destination picture
					*(dest + y * width + x) = ubyte(float(sum) / kernel_sum);
				}

			}
		}
	}
	*/
}

__constant__ int gpu_kernel_size;
__constant__ ubyte gpu_kernel;


__global__ void vignette(const unsigned char * src, unsigned char * dst, float inner, float outer, const size_t width, const size_t height)
{
	// the xIndex and yIndex will be used cordinates pixels of the image
	// NOTE
	// NOTE This assumes that we are treating this as a two dimensional data structure and the blocks will be used in the same way
	// NOTE
	size_t xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	size_t yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	// Checking to see if the indexs are within the bounds of the image
	if (xIndex < width && yIndex < height)
	{
		// offset represents postion of the current pixel in the one dimensional array
		size_t offset = yIndex * width + xIndex;
		// Shift the pixel oriented coordinates into image resolution independent coordinates
		// where 0, 0 is the center of the image.
		float x = xIndex / float(height) - float(width) / float(height) / 2.0f;
		float y = yIndex / float(height) - 0.5f;
		//Calculates current pixels distance from the center where the cordinates are 0, 0
		float d = sqrtf(x * x + y * y);
		if (d < inner)
		{
			// if d is less than inner boundary, we don't change that specific image pixel
			*(dst + offset) = *(src + offset);
		}
		else if (d > outer)
		{
			// if d is greater than outer boundary, we set it to 0 so it becomes black
			*(dst + offset) = 0;
		}
		else
		{
			// If in between the inner and outer boundaries, it will be a shade of gray.
			// NOTE
			// NOTE  This assumes... by the time we get here, we have checked that outer does not equal inner
			// NOTE  This also assumes ... by the time we get here, we have made inner less than outer
			// NOTE
			float v = 1 - (d - inner) / (outer - inner);
			*(dst + offset) = (unsigned char)(*(src + offset) * v);
		}
	}
}

__global__ void kernel(const unsigned char * src, unsigned char * dst, int level, const size_t width, const size_t height)
{
	const size_t xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const size_t yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	if (xIndex < width && yIndex < height)
	{
		size_t o = yIndex * width + xIndex;
		if (level == 256)
		{
			*(dst + o) = 0;
		}
		else
		{
			*(dst + o) = (*(src + o) >= level) ? 255 : 0;
		}
		// Notice how the below version avoids having an 'if' statement.
		// I wonder if this is truly correct - I'll have to test this
		// carefully someday but it works correctly. I figured the
		// subtraction should cause an underflow which the shift might
		// propagate through the rest of the byte so as to cause 255.
		// *(dst + o) = ~((*(src + o) - level - 1) >> 7);
	}
}

__global__ void box_filter_kernel(ubyte * src, ubyte * dest, int width, int height, ubyte * kernel, int kw, int kh) {
	//NOTE
	//NOTE	This assumes that we are working with a kernel that is a square and
	//NOTE	it has odd demensions (ex. 3x3)
	//NOTE 
	//grab the x and y values for the image on the gpu
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int y_index = blockIdx.y * blockDim.y + threadIdx.y;
	int kernel_size;
	

	kernel_size = kw * kh;
	const int half_kernel_size = kernel_size / 2;
	float kernel_sum;

	//get the total sum of the kernel to be used later as the divisor
	for (int i = 0; i < (kw * kh); i++) {
		kernel_sum += kernel[i];
	}
	//if the kernel sum is = 0, then assign it to be 1
	if (kernel_sum == 0) {
		kernel_sum = 1;
	}

	
	//check to see if the indexs are within the bounds of the image
	if (x_index < width && y_index < height) {
		//offset represents the position of the current pixel in the one dimensional array
		size_t offset = y_index * width + x_index;

		//reset the sum for the pixel we are currently working on to be 0
		int sum = 0;

		//points to the middle element of the row of the kernel
		//used to walk through each row of the kernel
		ubyte * kernel_p = kernel + half_kernel_size;

		//walk through the rows of the kernel
		for (int y_offset = -half_kernel_size; y_offset <= half_kernel_size; y_offset++, kernel_p += kernel_size) {
			//checks for the sides of the image
			if (y_offset + y_index < 0 || y_offset + y_index >= height) {
				continue;
			}

			//finds the value of the middle elements of the row
			sum += *(src + (y_offset + y_index) * width + x_index) * *kernel_p;

			//goes through the current row and finds the sums for the x values
			//to the side of the middle element of the row that kernel_p points to
			for (int offset_x = 1; offset_x <= half_kernel_size; offset_x++) {
				if (x_index - offset_x >= 0) {
					sum += *(src + (y_offset + y_index) * width - offset_x + x_index) * *(kernel_p - offset_x);
				}
				if (x_index + offset_x < width) {
					sum += *(src + (y_offset + y_index) * width + offset_x + x_index) * *(kernel_p + offset_x);
				}

				//assign the new values to the destination picture
				*(dest + y_index * width + x_index) = ubyte(float(sum) / kernel_sum);
			}

		}

	}
}


void on_trackbar(int, void *)
{
	dim3 grid((host_image.cols + 1023) / 1024, host_image.rows);
	double d;

	int i = inner_slider;
	int o = outer_slider;
	if (i > o)
	{
		swap(i, o);
	}
	if (i == o)
	{
		o++;
	}
	float inner = i / 100.0f;
	float outer = o / 100.0f;
	
	if (gpu_mode)
	{
		if (cudaMemcpy(device_src, orig_image.ptr(), image_bytes, cudaMemcpyHostToDevice) != cudaSuccess)
		{
			cerr << "cudaMemcpy failed at " << __LINE__ << endl;
			cudaDeviceReset();
			exit(1);
		}
	}
	else
	{
		host_image = orig_image;
		Threshold(host_image, threshold_slider);
		cout << "CPU AVG " << setw(12) << fixed << setprecision(8) << cpu_accumulator / ((double)cpu_counter) << " seconds";
		cout << endl;
	}




	imshow(window_name, host_image);
}

int main(int argc, char * argv[])
{
	//checks to see if there is a second argument
	if (argc != 2)
	{
		cout << " Usage: display_image ImageToLoadAndDisplay" << endl;
		return -1;
	}

	//reads in the image
	host_image = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	//checks to see if the image doesn't have any data
	if (!host_image.data)
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	cout << "Image has: " << host_image.channels() << " channels." << endl;
	cout << "Image has size: " << host_image.cols << " x " << host_image.rows << " pixels." << endl;

	//converts the image to grey
	cvtColor(host_image, host_image, cv::COLOR_RGB2GRAY);
	//copys oover the host image to the origional image
	host_image.copyTo(orig_image);
	cout << "Converted to gray." << endl;

	//sets the cuda device to the first one and checks to make sure it worked
	if (cudaSetDevice(0) != cudaSuccess)
	{
		cerr << "cudaSetDevice(0) failed." << endl;
		cudaDeviceReset();
		exit(1);
	}

	//gets the total amount of bytes in the image
	image_bytes = host_image.rows * host_image.cols;

	//mallocs space on the gpu for the source and destination images
	cudaMalloc(&device_src, image_bytes);
	cudaMalloc(&device_dst, image_bytes);
	//checks to make sure that the memory was allocated correctly
	if (device_dst == nullptr || device_src == nullptr)
	{
		cerr << "cudaMalloc failed on either device_src or device_dst at " << __LINE__ << endl;
		cudaDeviceReset();
		exit(1);
	}

	// Copy the source image to the device. Note - although the kernel will be
	// called repeatedly, this is the only time we'll copy TO the device as the
	// image processing operation does not harm the original image.
	if (cudaMemcpy(device_src, host_image.data, image_bytes, cudaMemcpyHostToDevice) != cudaSuccess)
	{
		cerr << "cudaMemcpy failed at " << __LINE__ << endl;
		cudaDeviceReset();
		exit(1);
	}

	/* for the cpu
	namedWindow(window_name, WINDOW_KEEPRATIO);
	resizeWindow(window_name, host_image.cols / 3, host_image.rows / 3);
	createTrackbar("kernel size", window_name, &blur_slider, THRESHOLD_SLIDER_MAX, on_trackbar);
	on_trackbar(blur_slider, 0);
	*/

	box_filter_kernel << < 1, 1024 >> > (device_src, device_dst, orig_image.cols, orig_image.rows, test_kernel, 3, 3);

	if (cudaMemcpy( final_image.data, device_src, image_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
		cerr << "cudaMemcpy failed at " << __LINE__ << endl;
	}

	namedWindow(window_name, WINDOW_KEEPRATIO);
	imshow(window_name, final_image);


	int k;

	while ((k = waitKey(10)) != 'q')
	{
		if (k == 'm')
		{
			gpu_mode = !gpu_mode;
			on_trackbar(0, nullptr);
		}
	}
	destroyAllWindows();

	if (device_src != nullptr)
		cudaFree(device_src);
	if (device_dst != nullptr)
		cudaFree(device_dst);
	cudaDeviceReset();
	return 0;
}