#include "MultibandBlender.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>

cv::Mat MultibandBlender::blend(const cv::Mat& img1, const cv::Mat& img2, const cv::Mat& mask)
{
    cv::Mat result;
    return result;
}


std::vector<cv::cuda::GpuMat> MultibandBlender::buildGaussianPyramid(const cv::cuda::GpuMat& img, int levels)
{
    cv::Mat result;
    return result;
}

std::vector<cv::cuda::GpuMat> MultibandBlender::buildLaplacianPyramid(const std::vector<cv::cuda::GpuMat>& gauss)
{
    cv::Mat result;
    return result;
}

std::vector<cv::cuda::GpuMat> MultibandBlender::blendPyramids(
    const std::vector<cv::cuda::GpuMat>& lap1,
    const std::vector<cv::cuda::GpuMat>& lap2,
    const std::vector<cv::cuda::GpuMat>& mask_gauss)
{
    cv::Mat result;
    return result;
}

cv::cuda::GpuMat MultibandBlender::reconstructFromLaplacianPyramid(const std::vector<cv::cuda::GpuMat>& lap)
{
    cv::cuda::GpuMat result;
    return result;
}