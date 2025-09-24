#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <vector>

class MultibandBlender
{
public:
    MultibandBlender(int levels = 5) : levels_(levels) {};

    // Blend two images with a mask
    cv::Mat blend(const cv::Mat& img1, const cv::Mat& img2, const cv::Mat& mask);
    std::vector<cv::cuda::GpuMat> buildGaussianPyramid(const cv::cuda::GpuMat& img, int levels, cv::cuda::Stream& s);
    std::vector<cv::cuda::GpuMat> buildLaplacianPyramid(const std::vector<cv::cuda::GpuMat>& gauss);

private:
    int levels_;

    std::vector<cv::cuda::GpuMat> blendPyramids(const std::vector<cv::cuda::GpuMat>& lap1, const std::vector<cv::cuda::GpuMat>& lap2, const std::vector<cv::cuda::GpuMat>& mask_gauss);
    cv::cuda::GpuMat reconstructFromLaplacianPyramid(const std::vector<cv::cuda::GpuMat>& lap);
};