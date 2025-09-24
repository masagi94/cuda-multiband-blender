#include "MultibandBlender.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>


// Combine the outputs gaussian and laplacian pyramids into a final blend
cv::Mat MultibandBlender::blend(const cv::Mat& left_img, const cv::Mat& right_img, const cv::Mat& mask)
{
    CV_Assert(left_img.size() == right_img.size() && left_img.type() == right_img.type());
    CV_Assert(mask.size() == left_img.size());
    CV_Assert(levels_ >= 1);

    // streams per input
    cv::cuda::Stream left_stream, right_stream, mask_stream;

    // upload
    cv::cuda::GpuMat left_img_gpu, right_img_gpu, mask_gpu;
    left_img_gpu.upload(left_img, left_stream);
    right_img_gpu.upload(right_img, right_stream);
    mask_gpu.upload(mask, mask_stream);

    // convert on gpu, normalize
    cv::cuda::GpuMat left_img_f_gpu, right_img_f_gpu, mask_f_gpu;
    left_img_gpu.convertTo(left_img_f_gpu, CV_32FC3, 1.0 / 255.0, 0.0, left_stream);
    right_img_gpu.convertTo(right_img_f_gpu, CV_32FC3, 1.0 / 255.0, 0.0, right_stream);
    if (mask_gpu.type() != CV_32FC1)
        mask_gpu.convertTo(mask_f_gpu, CV_32FC1, 1.0 / 255.0, 0.0, mask_stream);
    else
        mask_f_gpu = mask_gpu;

    // build 3 Gaussian pyramids in parallel
    auto left_gauss_pyrs = buildGaussianPyramid(left_img_f_gpu, levels_, left_stream);
    auto right_gauss_pyrs = buildGaussianPyramid(right_img_f_gpu, levels_, right_stream);
    auto mask_gauss_pyrs = buildGaussianPyramid(mask_f_gpu, levels_, mask_stream);

    // let streams finish
    left_stream.waitForCompletion();
    right_stream.waitForCompletion();
    mask_stream.waitForCompletion();


    auto left_lap_pyrs = buildLaplacianPyramid(left_gauss_pyrs);
    auto right_lap_pyrs = buildLaplacianPyramid(right_gauss_pyrs);

    // blend every pyramid level
    auto blended_lap_pyrs = blendPyramids(left_lap_pyrs, right_lap_pyrs, mask_gauss_pyrs);

    // add levels back together for final result
    cv::cuda::GpuMat blend_result_32_gpu = reconstructFromLaplacianPyramid(blended_lap_pyrs);

    cv::cuda::GpuMat blend_result_8_gpu;
    blend_result_32_gpu.convertTo(blend_result_8_gpu, CV_8UC3, 255.0);
    
    cv::Mat blend_result;
    blend_result_8_gpu.download(blend_result);

    return blend_result;
}

// Build gaussian pyramids by repeatedly smoothing and downsampling the image
std::vector<cv::cuda::GpuMat> MultibandBlender::buildGaussianPyramid(const cv::cuda::GpuMat& img, int levels, cv::cuda::Stream& s)
{
    std::vector<cv::cuda::GpuMat> gauss_pyrs(levels);
    gauss_pyrs[0] = img;
    for (int i = 0; i < levels - 1; i++) {
        cv::cuda::pyrDown(gauss_pyrs[i], gauss_pyrs[i + 1], s);
    }
    return gauss_pyrs;
}

// Build the laplacian pyramids by upsampling gauss pyr and subtracting the upsampled version from the original gauss
std::vector<cv::cuda::GpuMat> MultibandBlender::buildLaplacianPyramid(const std::vector<cv::cuda::GpuMat>& gauss_pyrs)
{
    const int pyr_num = static_cast<int>(gauss_pyrs.size());
    std::vector<cv::cuda::Stream> streams(pyr_num - 1);

    std::vector<cv::cuda::GpuMat> upsampled_pyrs(pyr_num - 1);
    std::vector<cv::cuda::GpuMat> lap_pyrs(pyr_num);

    for (int i = 0; i < pyr_num - 1; i++) {
        auto& s = streams[i];

        cv::cuda::pyrUp(gauss_pyrs[i + 1], upsampled_pyrs[i], s);

        // resize if needed
        if (upsampled_pyrs[i].size() != gauss_pyrs[i].size()) {
            cv::cuda::resize(upsampled_pyrs[i], upsampled_pyrs[i], gauss_pyrs[i].size(), 0, 0, cv::INTER_LINEAR, s);
        }

        cv::cuda::subtract(gauss_pyrs[i], upsampled_pyrs[i], lap_pyrs[i], cv::noArray(), -1, s);
    }

    // wait for streams to finish
    for (auto& s : streams) s.waitForCompletion();

    // last level is gaussian
    lap_pyrs[pyr_num - 1] = gauss_pyrs.back();

    return lap_pyrs;
}

// expand 1-ch mask to 3-ch
static void to3(const cv::cuda::GpuMat& one, cv::cuda::GpuMat& three) {
    std::vector<cv::cuda::GpuMat> ch(3, one);
    cv::cuda::merge(ch, three);
}

// Blend the pyramids by multiplying each level by their gradient mask, and summing the results
std::vector<cv::cuda::GpuMat> MultibandBlender::blendPyramids(const std::vector<cv::cuda::GpuMat>& left_lap_pyrs, const std::vector<cv::cuda::GpuMat>& right_lap_pyrs, const std::vector<cv::cuda::GpuMat>& mask_gauss)
{
    std::vector<cv::cuda::GpuMat> blended_pyrs;
    blended_pyrs.reserve(left_lap_pyrs.size());

    for (size_t i = 0; i < left_lap_pyrs.size(); i++) {
        
        cv::cuda::GpuMat ones(mask_gauss[i].size(), mask_gauss[i].type()); 
        ones.setTo(1.0);

        cv::cuda::GpuMat inverse_mask1; 
        cv::cuda::subtract(ones, mask_gauss[i], inverse_mask1);

        cv::cuda::GpuMat mask3, inverse_mask3; 
        to3(mask_gauss[i], mask3);
        to3(inverse_mask1, inverse_mask3);

        cv::cuda::GpuMat left_lap_scaled, right_lap_scaled, sum;
        cv::cuda::multiply(left_lap_pyrs[i], mask3, left_lap_scaled);
        cv::cuda::multiply(right_lap_pyrs[i], inverse_mask3, right_lap_scaled);
        cv::cuda::add(left_lap_scaled, right_lap_scaled, sum);
        blended_pyrs.push_back(sum);
    }
    return blended_pyrs;
}

// Reconstruct the final image by adding up all the pyramid levels
cv::cuda::GpuMat MultibandBlender::reconstructFromLaplacianPyramid(const std::vector<cv::cuda::GpuMat>& lap)
{
    cv::cuda::GpuMat cur_lap = lap.back();
    
    for (int i = static_cast<int>(lap.size()) - 2; i >= 0; i--) {
        cv::cuda::GpuMat upsampled_lap;
        cv::cuda::pyrUp(cur_lap, upsampled_lap);

        if (upsampled_lap.size() != lap[i].size())
            cv::cuda::resize(upsampled_lap, upsampled_lap, lap[i].size());
        
        cv::cuda::add(upsampled_lap, lap[i], cur_lap);
    }
    return cur_lap;
}