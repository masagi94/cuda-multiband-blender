#include "MultibandBlender.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>

cv::Mat MultibandBlender::blend(const cv::Mat& img1, const cv::Mat& img2, const cv::Mat& mask)
{
    cv::Mat result;
    return result;
}


std::vector<cv::cuda::GpuMat> MultibandBlender::buildGaussianPyramid(const cv::cuda::GpuMat& img, int levels)
{
    std::vector<cv::cuda::GpuMat> pyr;
    pyr.resize(levels);

    // level 0 is the original
    pyr[0] = img;

    for (int i = 0; i < levels - 1; ++i) {
        cv::cuda::pyrDown(pyr[i], pyr[i + 1]);
    }
    return pyr;
}

std::vector<cv::cuda::GpuMat> MultibandBlender::buildLaplacianPyramid(const std::vector<cv::cuda::GpuMat>& gauss)
{
    const int pyr_num = static_cast<int>(gauss.size());
    std::vector<cv::cuda::Stream> streams(pyr_num - 1);

    std::vector<cv::cuda::GpuMat> ups(pyr_num - 1);
    std::vector<cv::cuda::GpuMat> laps(pyr_num - 1);

    for (int i = 0; i < pyr_num - 1; i++) {
        auto& s = streams[i];

        cv::cuda::pyrUp(gauss[i + 1], ups[i], s);

        // resize if needed
        if (ups[i].size() != gauss[i].size()) {
            cv::cuda::resize(ups[i], ups[i], gauss[i].size(), 0, 0, cv::INTER_LINEAR, s);
        }

        cv::cuda::subtract(gauss[i], ups[i], laps[i], cv::noArray(), -1, s);
    }

    // wait for streams to complete
    for (auto& s : streams) s.waitForCompletion();

    std::vector<cv::cuda::GpuMat> lap_pyrs;
    lap_pyrs.reserve(pyr_num);
    for (int i = 0; i < pyr_num - 1; ++i) lap_pyrs.push_back(laps[i]);
    
    // last level is gaussian
    lap_pyrs.push_back(gauss.back());
    return lap_pyrs;
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