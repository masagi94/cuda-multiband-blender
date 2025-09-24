#include "MultibandBlender.hpp"


cv::Mat MultibandBlender::blend(const cv::Mat& img1, const cv::Mat& img2, const cv::Mat& mask)
{
    cv::Mat result;
    return result;
}


std::vector<cv::cuda::GpuMat> MultibandBlender::build_gaussian_pyramid_cuda(const cv::cuda::GpuMat& img, int levels)
{
    cv::Mat result;
    return result;
}

std::vector<cv::cuda::GpuMat> MultibandBlender::build_laplacian_pyramid_cuda(const std::vector<cv::cuda::GpuMat>& gauss)
{
    cv::Mat result;
    return result;
}

std::vector<cv::cuda::GpuMat> MultibandBlender::blend_pyramids_cuda(
    const std::vector<cv::cuda::GpuMat>& lap1,
    const std::vector<cv::cuda::GpuMat>& lap2,
    const std::vector<cv::cuda::GpuMat>& mask_gauss)
{
    cv::Mat result;
    return result;
}

cv::cuda::GpuMat MultibandBlender::reconstruct_from_laplacian_pyramid_cuda(const std::vector<cv::cuda::GpuMat>& lap)
{
    cv::cuda::GpuMat result;
    return result;
}