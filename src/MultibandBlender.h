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

private:
    int levels_;

    // CUDA helper functions
    std::vector<cv::cuda::GpuMat> build_gaussian_pyramid_cuda(const cv::cuda::GpuMat& img, int levels);
    std::vector<cv::cuda::GpuMat> build_laplacian_pyramid_cuda(const std::vector<cv::cuda::GpuMat>& gauss);
    std::vector<cv::cuda::GpuMat> blend_pyramids_cuda(const std::vector<cv::cuda::GpuMat>& lap1, 
                                                      const std::vector<cv::cuda::GpuMat>& lap2, 
                                                      const std::vector<cv::cuda::GpuMat>& mask_gauss);
    cv::cuda::GpuMat reconstruct_from_laplacian_pyramid_cuda(const std::vector<cv::cuda::GpuMat>& lap);
};