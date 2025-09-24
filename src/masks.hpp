#pragma once
#include <opencv2/core.hpp>

namespace blend::masks {

    // Build complementary 8-bit masks (0..255) using a cosine ramp
    void makeSoftMasks(const cv::Mat& img_left, const cv::Mat& img_right, cv::Mat& mask_left, cv::Mat& mask_right, int overlap_cols = 50);

}