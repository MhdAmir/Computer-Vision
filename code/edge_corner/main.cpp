#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <chrono>

class RobustCornerDetector {
private:
    int threshold;
    bool nonmaxSuppression;
    cv::Ptr<cv::FastFeatureDetector> fastDetector;
    cv::Ptr<cv::BFMatcher> matcher;
    cv::Ptr<cv::BRISK> brisk;
    
    const int MAX_FEATURES = 500;
    const float RATIO_THRESHOLD = 0.7f;
    const int MIN_MATCHES = 10;  
    
public:
    RobustCornerDetector(int threshold = 40, bool nonmaxSuppression = true) 
        : threshold(threshold), nonmaxSuppression(nonmaxSuppression) {
        fastDetector = cv::FastFeatureDetector::create(threshold, nonmaxSuppression);
        matcher = cv::BFMatcher::create(cv::NORM_HAMMING, false);
        brisk = cv::BRISK::create();
    }
    
    struct ProcessingResult {
        std::vector<cv::KeyPoint> keypoints;
        double processingTime;
        cv::Mat outputImage;
    };
    
    ProcessingResult processFrame(const cv::Mat& frame) {
        auto startTime = std::chrono::high_resolution_clock::now();
        ProcessingResult result;
        
        if (frame.empty()) {
            std::cout << "Warning: Empty frame received" << std::endl;
            return result;
        }
        
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
        
        result.outputImage = frame.clone();
        
        for (const auto& kp : keypoints) {
            cv::circle(result.outputImage, kp.pt, 3, cv::Scalar(0, 255, 0), 2);
            std::string str = std::to_string(static_cast<int>(kp.response));
            cv::putText(result.outputImage, str, kp.pt + cv::Point2f(5, 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        }
        
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        
        result.keypoints = keypoints;
        result.processingTime = duration.count();
        
        return result;
    }
    
    cv::Mat getFeatureMatchingVisual(const cv::Mat& frame1, const cv::Mat& frame2) {
        if (frame1.empty() || frame2.empty()) {
            std::cout << "Warning: Empty frame(s) received for matching" << std::endl;
            return cv::Mat();
        }
        
        cv::Mat gray1, gray2;
        if (frame1.channels() == 3) cv::cvtColor(frame1, gray1, cv::COLOR_BGR2GRAY);
        else gray1 = frame1.clone();
        if (frame2.channels() == 3) cv::cvtColor(frame2, gray2, cv::COLOR_BGR2GRAY);
        else gray2 = frame2.clone();
        
        std::vector<cv::KeyPoint> keypoints1, keypoints2;
        cv::Mat descriptors1, descriptors2;
        
        brisk->detectAndCompute(gray1, cv::noArray(), keypoints1, descriptors1);
        brisk->detectAndCompute(gray2, cv::noArray(), keypoints2, descriptors2);
        
        if (keypoints1.empty() || keypoints2.empty() || 
            descriptors1.empty() || descriptors2.empty()) {
            std::cout << "Warning: Not enough keypoints detected" << std::endl;
            return frame2.clone(); 
        }
        
        std::vector<std::vector<cv::DMatch>> knnMatches;
        try {
            matcher->knnMatch(descriptors1, descriptors2, knnMatches, 2);
        } catch (const cv::Exception& e) {
            std::cout << "Warning: Matching failed - " << e.what() << std::endl;
            return frame2.clone();
        }
        
        std::vector<cv::DMatch> goodMatches;
        for (const auto& match : knnMatches) {
            if (match.size() >= 2 && 
                match[0].distance < RATIO_THRESHOLD * match[1].distance) {
                goodMatches.push_back(match[0]);
            }
        }
        
        if (goodMatches.size() < MIN_MATCHES) {
            std::cout << "Warning: Not enough good matches found (" 
                     << goodMatches.size() << " < " << MIN_MATCHES << ")" << std::endl;
            return frame2.clone();
        }
        
        cv::Mat imgMatches;
        try {
            cv::drawMatches(frame1, keypoints1, frame2, keypoints2, goodMatches, imgMatches,
                           cv::Scalar::all(-1), cv::Scalar::all(-1),
                           std::vector<char>(), cv::DrawMatchesFlags::DEFAULT);
        } catch (const cv::Exception& e) {
            std::cout << "Warning: Drawing matches failed - " << e.what() << std::endl;
            return frame2.clone();
        }
        
        return imgMatches;
    }
};

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cout << "Error opening camera" << std::endl;
        return -1;
    }
    
    RobustCornerDetector detector(40);
    cv::Mat prevFrame;
    bool isFirstFrame = true;
    
    while (true) {
        cv::Mat frame;
        cap >> frame;
        
        if (frame.empty()) {
            std::cout << "Error: Blank frame" << std::endl;
            break;
        }
        
        auto result = detector.processFrame(frame);
        
        std::string timeText = "Processing time: " + std::to_string(result.processingTime) + "ms";
        cv::putText(result.outputImage, timeText, cv::Point(10, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        
        std::string cornerText = "Corners: " + std::to_string(result.keypoints.size());
        cv::putText(result.outputImage, cornerText, cv::Point(10, 60),
                   cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        
        if (!isFirstFrame) {
            cv::Mat matchingVisual = detector.getFeatureMatchingVisual(prevFrame, frame);
            if (!matchingVisual.empty()) {
                cv::imshow("Feature Matching", matchingVisual);
            }
        }
        
        frame.copyTo(prevFrame);
        isFirstFrame = false;
        
        cv::imshow("Robust Corner Detection", result.outputImage);
        
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }
    
    cap.release();
    cv::destroyAllWindows();
    return 0;
}