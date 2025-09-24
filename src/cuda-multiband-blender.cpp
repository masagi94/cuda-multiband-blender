#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>

#include "MultibandBlender.hpp"
#include "masks.hpp"

namespace fs = std::filesystem;


// Check if cuda is available. Need it to run the program
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

// Use OpenCV's feather blender on the input images
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

// Save gauss and lap pyramids for evaluation
static void saveGaussLapPyramids(MultibandBlender& custom_blender, cv::Mat& left_img, cv::Mat& right_img, cv::Mat& left_mask, fs::path& pyr_dir) {
    cv::cuda::GpuMat left_img_gpu, left_mask_gpu;
    left_img_gpu.upload(left_img);
    left_mask_gpu.upload(left_mask);
    cv::cuda::Stream s;

    // save left image gauss pyramids
    auto pyr_results = custom_blender.buildGaussianPyramid(left_img_gpu, 5, s);
    for (int i = 0; i < pyr_results.size(); i++) {
        fs::path out_path = pyr_dir / ("gauss_" + std::to_string(i) + ".png");
        cv::Mat temp;
        pyr_results[i].download(temp);
        //cv::imshow(std::to_string(i), temp);
        cv::imwrite(out_path.string(), temp);
    }

    
    // save left image laplace pyramids
    auto laplace_result = custom_blender.buildLaplacianPyramid(pyr_results);
    for (int i = 0; i < laplace_result.size(); i++) {
        fs::path out_path = pyr_dir / ("lap_" + std::to_string(i) + ".png");
        cv::Mat temp;
        laplace_result[i].download(temp);
        //cv::imshow(std::to_string(i), temp);
        cv::imwrite(out_path.string(), temp);

    }

    // save left mask gauss pyramids
    auto mask_pyr_results = custom_blender.buildGaussianPyramid(left_mask_gpu, 5, s);
    for (int i = 0; i < mask_pyr_results.size(); i++) {
        fs::path out_path = pyr_dir / ("gauss_mask_" + std::to_string(i) + ".png");
        cv::Mat temp;
        mask_pyr_results[i].download(temp);
        //cv::imshow(std::to_string(i), temp);
        cv::imwrite(out_path.string(), temp);
    }

    std::cout << "Saved pyramids to: " << pyr_dir << "\n";
}



int main()
{
    bool cuda_available = checkForCuda();
    if (!cuda_available) {
        return EXIT_FAILURE;
    }
    
    // Set up paths for inputs and outputs 
    fs::path data_dir = "data";    
    fs::path left_img_path = data_dir / "f16_left.png";
    fs::path right_img_path = data_dir / "f16_right.png";
    //fs::path right_p = data_dir / "data/goose.png";

    fs::path results_dir = "results";
    fs::create_directories(results_dir);
    fs::path multiband_result_path = results_dir / "multiband_blend.png";
    fs::path feather_result_path = results_dir / "feather_blend.png";
    fs::path mask_result_path = results_dir / "mask.png";
    fs::path pyr_dir = results_dir / "pyramids";
    fs::create_directories(pyr_dir);



    // Load images, check size 
    cv::Mat left_img = loadImage(left_img_path);
    cv::Mat right_img = loadImage(right_img_path);
    
    if (left_img.size() != right_img.size()) {
        cv::resize(right_img, right_img, left_img.size());
    }
    
    // Create masks
    cv::Mat left_mask, right_mask;
    const int overlap_cols = 20;
    blend::masks::makeSoftMasks(left_img, right_img, left_mask, right_mask, overlap_cols);

    // Create ROI
    const int width = left_img.cols;
    const int height = left_img.rows;
    cv::Rect roi_location(cv::Point(0, 0), cv::Size(width, height));

    // Brighten one image to make blending more difficult
    const double alpha = 1.5;
    const double beta = 0;
    right_img.convertTo(right_img, -1, alpha, beta);


    // Run custom blender, save output
    MultibandBlender custom_blender(8);
    cv::Mat overlap_blended = custom_blender.blend(left_img, right_img, left_mask);
    saveGaussLapPyramids(custom_blender, left_img, right_img, left_mask, pyr_dir);
    cv::imwrite(multiband_result_path.string(), overlap_blended);
    cv::imshow("Custom blender", overlap_blended);
    
    // Prepare inputs for Feather Blender
    cv::Mat left_img_16s, right_img_16s;
    left_img.convertTo(left_img_16s, CV_16SC3);
    right_img.convertTo(right_img_16s, CV_16SC3);

    // Run feather blender, save output
    cv::Mat feather_output = runFeatherBlend(left_img_16s, right_img_16s, left_mask, right_mask, roi_location);
    cv::imwrite(feather_result_path.string(), feather_output);
    cv::imwrite(mask_result_path.string(), left_mask);
    cv::imshow("OpenCV Feather", feather_output);
    cv::waitKey(0);
}


