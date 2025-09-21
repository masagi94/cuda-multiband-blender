
#include <iostream>
#include <opencv2/opencv.hpp>
#include "MultibandBlender.h"

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












