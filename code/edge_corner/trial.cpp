#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <chrono>

int main() {
    // Initialize camera
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cout << "Error opening camera" << std::endl;
        return -1;
    }

    const int FAST_THRESHOLD = 40;
    const bool NONMAX_SUPPRESSION = true;
    const int MAX_FEATURES = 500;
    const float RATIO_THRESHOLD = 0.7f;
    const int MIN_MATCHES = 10;

    auto fastDetector = cv::FastFeatureDetector::create(FAST_THRESHOLD, NONMAX_SUPPRESSION);
    auto matcher = cv::BFMatcher::create(cv::NORM_HAMMING, false);
    auto brisk = cv::BRISK::create();

    cv::Mat prevFrame;
    bool isFirstFrame = true;

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) {
            std::cout << "Error: Blank frame" << std::endl;
            break;
        }

        auto startTime = std::chrono::high_resolution_clock::now();

        cv::Mat grayFrame;
        if (frame.channels() == 3) {
            cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
        } else {
            grayFrame = frame.clone();
        }

        std::vector<cv::KeyPoint> keypoints;
        fastDetector->detect(grayFrame, keypoints);

        if (keypoints.size() > MAX_FEATURES) {
            std::sort(keypoints.begin(), keypoints.end(),
                     [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
                         return a.response > b.response;
                     });
            keypoints.resize(MAX_FEATURES);
        }

        cv::Mat outputImage = frame.clone();
        for (const auto& kp : keypoints) {
            cv::circle(outputImage, kp.pt, 3, cv::Scalar(0, 255, 0), 2);
            std::string str = std::to_string(static_cast<int>(kp.response));
            cv::putText(outputImage, str, kp.pt + cv::Point2f(5, 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        std::string timeText = "Processing time: " + std::to_string(duration.count()) + "ms";
        cv::putText(outputImage, timeText, cv::Point(10, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

        std::string cornerText = "Corners: " + std::to_string(keypoints.size());
        cv::putText(outputImage, cornerText, cv::Point(10, 60),
                   cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

        if (!isFirstFrame && !prevFrame.empty()) {
            try {
                std::vector<cv::KeyPoint> kp1, kp2;
                cv::Mat desc1, desc2;
                
                brisk->detectAndCompute(prevFrame, cv::noArray(), kp1, desc1);
                brisk->detectAndCompute(frame, cv::noArray(), kp2, desc2);

                if (!kp1.empty() && !kp2.empty() && !desc1.empty() && !desc2.empty()) {
                    std::vector<std::vector<cv::DMatch>> knnMatches;
                    matcher->knnMatch(desc1, desc2, knnMatches, 2);

                    std::vector<cv::DMatch> goodMatches;
                    for (const auto& match : knnMatches) {
                        if (match.size() >= 2 && 
                            match[0].distance < RATIO_THRESHOLD * match[1].distance) {
                            goodMatches.push_back(match[0]);
                        }
                    }

                    if (goodMatches.size() >= MIN_MATCHES) {
                        cv::Mat matchingVisual;
                        cv::drawMatches(prevFrame, kp1, frame, kp2, goodMatches, matchingVisual,
                                      cv::Scalar::all(-1), cv::Scalar::all(-1),
                                      std::vector<char>(), cv::DrawMatchesFlags::DEFAULT);
                        cv::imshow("Feature Matching", matchingVisual);
                    }
                }
            } catch (const cv::Exception& e) {
                std::cout << "Warning: Matching failed - " << e.what() << std::endl;
            }
        }

        frame.copyTo(prevFrame);
        isFirstFrame = false;

        cv::imshow("Corner Detection", outputImage);

        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}