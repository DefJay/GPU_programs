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

void box_filter(ubyte * source, ubyte * dest, int width, int height, ubyte * kernel, int kw, int kh, ubyte * temp) {
	int kernel_sum = 0;
	int sum = 0;
	//find the sum of all the elements in the kernel
	for (int i = 0; i < kw; i++) {
		for (int j = 0; j < kh; j++) {
			kernel_sum = kernel_sum + kernel[i + j];
		}
	}
	cout << "kernel's total sum: " << kernel_sum << endl;

	//clone the origional image's data to the temp and dest
	temp = source;
	dest = source;


	//if the kernel sum is = 0, assign it to be 1
	if (kernel_sum == 0) {
		kernel_sum = 1;
	}

	//bluring for a 3x3 matrix
	for (int y = 1; y < height - 1; y++) {
		for (int x = 1; x < width - 1; x++) {
			ubyte new_val = source[one_deminsional(x - 1, y - 1, width)] * kernel[0];
			new_val += source[one_deminsional(x , y - 1, width)] * kernel[1];
			new_val += source[one_deminsional(x + 1, y - 1, width)] * kernel[2];
			new_val += source[one_deminsional(x - 1 , y, width)] * kernel[3];
			new_val += source[one_deminsional(x, y, width)] * kernel[4];
			new_val += source[one_deminsional(x + 1, y, width)] * kernel[5];
			new_val += source[one_deminsional(x - 1, y + 1, width)] * kernel[6];
			new_val += source[one_deminsional(x, y + 1, width)] * kernel[7];
			new_val += source[one_deminsional(x + 1, y + 1, width)] * kernel[8];

			new_val = new_val / kernel_sum;

			temp[one_deminsional(x, y, width)] = new_val;

		}

		dest = temp;
	}


	/*
	//go through every row and every column of the image
	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {
			//reset the sum to 0
			sum = 0;

			//go through every row & column of the kernel
			for (int kernel_x = 0; kernel_x < kw; kernel_x++) {
				for (int kernel_y = 0; kernel_y < kh; kernel_y++) {
					
					
				}
			}
			

		}
	}
	

	//assign the new values
	for (int i = 0; i < (width * height); i++) {
		dest[i] = temp[i];
	}
	*/
}




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
	/*
	if (gpu_mode)
	{
		if (cudaMemcpy(device_src, orig_image.ptr(), image_bytes, cudaMemcpyHostToDevice) != cudaSuccess)
		{
			cerr << "cudaMemcpy failed at " << __LINE__ << endl;
			cudaDeviceReset();
			exit(1);
		}

		t.TimeSinceLastCall();
		kernel << <grid, 1024 >> >(device_src, device_dst, threshold_slider, host_image.cols, host_image.rows);
		cudaDeviceSynchronize();
		gpu_accumulator += t.TimeSinceLastCall();
		gpu_counter++;
		cout << "GPU AVG " << setw(12) << fixed << setprecision(8) << gpu_accumulator / ((double)gpu_counter) << " seconds";
		vignette << <grid, 1024 >> >(device_dst, device_src, inner, outer, host_image.cols, host_image.rows);
		cudaDeviceSynchronize();

		t.TimeSinceLastCall();
		if (cudaMemcpy(host_image.ptr(), device_src, image_bytes, cudaMemcpyDeviceToHost) != cudaSuccess)
		{
			cerr << "cudaMemcpy failed at " << __LINE__ << endl;
		}
		d = t.TimeSinceLastCall();
		cout << " XFER: " << setw(8) << setprecision(4) << ((double)image_bytes) / (d * 1024.0 * 1024.0 * 1024.0) << " GB/s" << endl;
	}
	else
	{
		t.TimeSinceLastCall();
		host_image = orig_image;
		Threshold(host_image, threshold_slider);
		cpu_accumulator += t.TimeSinceLastCall();
		cpu_counter++;
		cout << "CPU AVG " << setw(12) << fixed << setprecision(8) << cpu_accumulator / ((double)cpu_counter) << " seconds";
		cout << endl;
	}
	*/



	imshow(window_name, host_image);
}

int main(int argc, char * argv[])
{
	if (argc != 2)
	{
		cout << " Usage: display_image ImageToLoadAndDisplay" << endl;
		return -1;
	}
	host_image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	if (!host_image.data)
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	cout << "Image has: " << host_image.channels() << " channels." << endl;
	cout << "Image has size: " << host_image.cols << " x " << host_image.rows << " pixels." << endl;

	cvtColor(host_image, host_image, cv::COLOR_RGB2GRAY);
	host_image.copyTo(orig_image);
	cout << "Converted to gray." << endl;

	/*
	if (cudaSetDevice(0) != cudaSuccess)
	{
		cerr << "cudaSetDevice(0) failed." << endl;
		cudaDeviceReset();
		exit(1);
	}
	image_bytes = host_image.rows * host_image.cols;
	cudaMalloc(&device_src, image_bytes);
	cudaMalloc(&device_dst, image_bytes);
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
	*/
	//namedWindow(window_name, WINDOW_KEEPRATIO);
	//resizeWindow(window_name, host_image.cols / 10, host_image.rows / 10);
	//createTrackbar("Threshold", window_name, &threshold_slider, THRESHOLD_SLIDER_MAX, on_trackbar);
	//createTrackbar("Inner", window_name, &inner_slider, 100, on_trackbar);
	//createTrackbar("Outer", window_name, &outer_slider, 100, on_trackbar);
	//on_trackbar(threshold_slider, 0);

	//copy over the host image to the final image & temp image
	final_image = host_image;
	temp_image = host_image;

	namedWindow("ORIGIONAL IMAGE", WINDOW_KEEPRATIO);
	resizeWindow("ORIGIONAL IMAGE", host_image.cols / 3, host_image.rows / 3);
	imshow("ORIGIONAL IMAGE", host_image);
	waitKey(0);

	box_filter(host_image.data, final_image.data, host_image.cols, host_image.rows, test_kernel, 3, 3, temp_image.data);

	
	namedWindow("FINAL IMAGE", WINDOW_KEEPRATIO);
	resizeWindow("FINAL IMAGE", host_image.cols / 3, host_image.rows / 3);
	imshow("FINAL IMAGE", final_image);
	waitKey(0);







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