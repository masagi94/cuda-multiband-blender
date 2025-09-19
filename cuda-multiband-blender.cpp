// cuda-multiband-blender.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/stitching/detail/warpers.hpp>
#include <opencv2/stitching/detail/matchers.hpp>
#include <opencv2/cudaarithm.hpp>

int main()
{
    std::cout << "hello world! gonna blend some stuff later\n\n";

    int n = cv::cuda::getCudaEnabledDeviceCount();
    std::cout << "CUDA devices seen by OpenCV: " << n << "\n";
    if (n > 0) {
        cv::cuda::DeviceInfo info(0);
        std::cout << "Using device 0: " << info.name() << "\n";
    }
    else {
        std::cout << "No CUDA device detected (or CUDA disabled in your OpenCV build).\n";
    }

}












