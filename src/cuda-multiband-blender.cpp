
#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>

#include "MultibandBlender.h"

namespace fs = std::filesystem;

static void buildCosineMasks(int width, int height, int start_col, int end_col, cv::Mat& mask_left, cv::Mat& mask_right) {
    
    mask_left.create(height, width, CV_8UC1);
    const int span = std::max(1, end_col - start_col);

    for (int y = 0; y < height; ++y) {

        uchar* mask_row = mask_left.ptr<uchar>(y);
        
        for (int x = 0; x < width; ++x) {
            if (x < start_col) {
                mask_row[x] = 255;
            }
            else if (x > end_col) {
                mask_row[x] = 0;
            }

            // apply gradient to the overlap
            else {
                double progress = double(x - start_col) / span;               
                double left_weight = 0.5 + 0.5 * std::cos(CV_PI * progress);
                
                mask_row[x] = (uchar)cv::saturate_cast<int>(255.0 * left_weight);
            }
        }
    }

    // left and right masks need to be complements
    cv::bitwise_not(mask_left, mask_right);
}

// Returns the midpoint between the last non-black col of left image and first non-black of the right image.
static int findSeamCenterCol(const cv::Mat& img_left, const cv::Mat& img_right, int non_black_threshold = 1) {
    
    CV_Assert(img_left.size() == img_right.size() && img_left.type() == CV_8UC3 && img_right.type() == CV_8UC3);
    const int width = img_left.cols;

    cv::Mat left_gray, right_gray;
    cv::cvtColor(img_left, left_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img_right, right_gray, cv::COLOR_BGR2GRAY);

    // find last non-black in left (scanning right -> left)
    int left_edge = -1;
    for (int x = width - 1; x >= 0; --x) {
        if (cv::countNonZero(left_gray.col(x) > non_black_threshold) > 0) {
            left_edge = x;
            break; 
        }
    }

    // find first non-black in right (scanning left -> right)
    int right_edge = -1;
    for (int x = 0; x < width; ++x) {
        if (cv::countNonZero(right_gray.col(x) > non_black_threshold) > 0) {
            right_edge = x;
            break; 
        }
    }
    
    // if either pic is all black, return mid
    if (left_edge < 0 || right_edge < 0) return width / 2;           
    return (left_edge + right_edge) / 2;
}

// Make soft gradient masks with a configurable overlap width.
static void makeSoftMasks(const cv::Mat& img_left, const cv::Mat& img_right, cv::Mat& mask_left, cv::Mat& mask_right, int overlap_cols = 50) {
    
    CV_Assert(img_left.size() == img_right.size());
    const int width = img_left.cols, height = img_left.rows;

    const int center_col = findSeamCenterCol(img_left, img_right);
    const int start_col = std::max(0, center_col - (overlap_cols / 2));
    const int end_col = std::min(width - 1, center_col + (overlap_cols / 2));

    buildCosineMasks(width, height, start_col, end_col, mask_left, mask_right);
}

static void checkForCuda() {
    std::cout << "hello world! gonna blend some stuff later\n\n";

    int n = cv::cuda::getCudaEnabledDeviceCount();
    std::cout << "CUDA devices seen by OpenCV: " << n << "\n";
    if (n > 0) {
        cv::cuda::DeviceInfo info(0);
        std::cout << "Using device 0: " << info.name() << "\n";
    }
    else {
        std::cout << "No CUDA device detected. Terminating project...\n";
        exit(0);
    }
}

// Loads input images
static cv::Mat loadImage(const fs::path& image_path) {
    
    cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);

    if (image.empty())
        CV_Error(cv::Error::StsError, cv::format("\n\nFailed to load: %s", image_path.string().c_str()));
    
    return image;
}


static cv::Mat testFeatherBlend(cv::Mat& left_16s, cv::Mat& right_16s, cv::Mat& mask_left, cv::Mat& mask_right, cv::Rect& roi_location) {
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
    checkForCuda();

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
    makeSoftMasks(left_img, right_img, mask_left, mask_right, overlap_cols);

    /* Create ROI */
    const int width = left_img.cols;
    const int height = left_img.rows;
    cv::Rect roi_location(cv::Point(0, 0), cv::Size(width, height));

    /* Prepare inputs for blending */
    const double alpha = 1.5;
    const double beta = 0;
    right_img.convertTo(right_img, -1, alpha, beta);

    cv::Mat left_img_16s, right_img_16s;
    left_img.convertTo(left_img_16s, CV_16SC3);
    right_img.convertTo(right_img_16s, CV_16SC3);

    cv::Mat feather_output = testFeatherBlend(left_img_16s, right_img_16s, mask_left, mask_right, roi_location);

    cv::imshow("OpenCV Feather", feather_output);
    cv::waitKey(0);

    cv::imwrite(feather_result_path.string(), feather_output);
    cv::imwrite(mask_result_p.string(), mask_left);
}












