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
    std::vector<cv::cuda::GpuMat> pyr;
    pyr.reserve(levels);

    cv::cuda::GpuMat cur_img = img;
    pyr.push_back(cur_img);
    
    for (int i = 1; i < levels; ++i) {
        cv::cuda::GpuMat down;
        cv::cuda::pyrDown(cur_img, down);
        pyr.push_back(down);
        cur_img = down;
    }
    return pyr;
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