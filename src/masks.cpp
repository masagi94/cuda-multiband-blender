#include "masks.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>

namespace blend::masks {

    // Builds the masks and applies a smooth gradient from left to right, and right to left
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
    void makeSoftMasks(const cv::Mat& img_left, const cv::Mat& img_right, cv::Mat& mask_left, cv::Mat& mask_right, int overlap_cols = 50) {

        CV_Assert(img_left.size() == img_right.size());
        const int width = img_left.cols, height = img_left.rows;

        const int center_col = findSeamCenterCol(img_left, img_right);
        const int start_col = std::max(0, center_col - (overlap_cols / 2));
        const int end_col = std::min(width - 1, center_col + (overlap_cols / 2));

        buildCosineMasks(width, height, start_col, end_col, mask_left, mask_right);
    }
}