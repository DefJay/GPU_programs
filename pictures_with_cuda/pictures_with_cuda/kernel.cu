#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;

void Threshold(int threashold, int width, int height, unsigned char * data);

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


	Threshold(178, image.cols, image.rows, image.data);

	namedWindow("Display window", WINDOW_NORMAL);
	imshow("Display window", image);

	waitKey(0);

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
