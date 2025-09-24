
#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>

#include "MultibandBlender.hpp"
#include "masks.hpp"

namespace fs = std::filesystem;



static bool checkForCuda() {

    int n = cv::cuda::getCudaEnabledDeviceCount();
    std::cout << "CUDA devices seen by OpenCV: " << n << "\n";
    if (n > 0) {
        cv::cuda::DeviceInfo info(0);
        std::cout << "Using device 0: " << info.name() << "\n";
        return 1;
    }
    else {
        std::cout << "No CUDA device detected.\n";
        return 0;
    }
}

// Loads input images
static cv::Mat loadImage(const fs::path& image_path) {
    
    cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);

    if (image.empty())
        CV_Error(cv::Error::StsError, cv::format("\n\nFailed to load: %s", image_path.string().c_str()));
    
    return image;
}


static cv::Mat runFeatherBlend(const cv::Mat& left_16s, const cv::Mat& right_16s, const cv::Mat& mask_left, const cv::Mat& mask_right, const cv::Rect& roi_location) {
    cv::detail::FeatherBlender feather_blender;
    
    feather_blender.prepare(roi_location);
    feather_blender.feed(left_16s, mask_left, cv::Point(0, 0));
    feather_blender.feed(right_16s, mask_right, cv::Point(0, 0));

    cv::Mat feather_output, feather_mask;
    
    feather_blender.blend(feather_output, feather_mask);
    
    cv::Mat feather_output_8U;
    feather_output.convertTo(feather_output_8U, CV_8U);

    return feather_output_8U;
}



int main()
{
    bool cuda_available = checkForCuda();
    if (!cuda_available) {
        return EXIT_FAILURE;
    }
    

    /* Set up paths for inputs and outputs */
    fs::path data_dir = "data";    
    fs::path left_img_path = data_dir / "f16_left.png";
    fs::path right_img_path = data_dir / "f16_right.png";
    //fs::path right_p = data_dir / "data/goose.png";

    fs::path results_dir = "results";
    fs::create_directories(results_dir);
    fs::path multiband_result_path = results_dir / "multiband_blend.png";
    fs::path feather_result_path = results_dir / "feather_blend.png";
    fs::path mask_result_p = results_dir / "mask.png";


    /* Load images, check size */
    cv::Mat left_img = loadImage(left_img_path);
    cv::Mat right_img = loadImage(right_img_path);
    
    if (left_img.size() != right_img.size()) {
        cv::resize(right_img, right_img, left_img.size());
    }
    
    /* Create masks */
    cv::Mat mask_left, mask_right;
    const int overlap_cols = 100;
    blend::masks::makeSoftMasks(left_img, right_img, mask_left, mask_right, overlap_cols);

    /* Create ROI */
    const int width = left_img.cols;
    const int height = left_img.rows;
    cv::Rect roi_location(cv::Point(0, 0), cv::Size(width, height));

    /* Prepare inputs for blending */
    const double alpha = 1.5;
    const double beta = 0;
    right_img.convertTo(right_img, -1, alpha, beta);



    MultibandBlender custom_blender;

    cv::cuda::GpuMat left_img_gpu;
    std::vector< cv::cuda::GpuMat> pyr_results;
    left_img_gpu.upload(left_img);
    pyr_results = custom_blender.buildGaussianPyramid(left_img_gpu, 5);

    std::cout << "*** pyr size: " << pyr_results.size();

    for (int i = 0; i < pyr_results.size(); i++) {
        cv::Mat temp;
        pyr_results[i].download(temp);
        cv::imshow(std::to_string(i), temp);
    }
    cv::waitKey(0);

    /*auto laplac_result = custom_blender.buildLaplacianPyramid(pyr_results);
    
    std::cout << "*** pyr size: " << laplac_result.size();

    for (int i = 0; i < laplac_result.size(); i++) {
        cv::Mat temp;
        laplac_result[i].download(temp);
        cv::imshow(std::to_string(i), temp);
    }
    cv::waitKey(0);*/




    cv::Mat left_img_16s, right_img_16s;
    left_img.convertTo(left_img_16s, CV_16SC3);
    right_img.convertTo(right_img_16s, CV_16SC3);

    cv::Mat feather_output = runFeatherBlend(left_img_16s, right_img_16s, mask_left, mask_right, roi_location);

    cv::imshow("OpenCV Feather", feather_output);
    cv::waitKey(0);

    cv::imwrite(feather_result_path.string(), feather_output);
    cv::imwrite(mask_result_p.string(), mask_left);
}












