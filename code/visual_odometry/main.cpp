#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

class VisualOdometry {
private:
    cv::Mat K;  // Camera intrinsic matrix
    cv::Mat prev_frame;
    cv::Mat curr_frame;
    std::vector<cv::Point2f> prev_points;
    std::vector<cv::Point2f> curr_points;
    cv::Mat R_f, t_f; // Final rotation and translation
    bool is_initialized;
    
    const int MIN_FEATURES = 1500;
    const double MIN_MATCH_DIST = 30.0;

public:
    VisualOdometry(const cv::Mat& camera_matrix) : K(camera_matrix), is_initialized(false) {
        R_f = cv::Mat::eye(3, 3, CV_64F);
        t_f = cv::Mat::zeros(3, 1, CV_64F);
    }

    void detectFeatures(const cv::Mat& frame, std::vector<cv::Point2f>& points) {
        std::vector<cv::KeyPoint> keypoints;
        cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create(20);
        detector->detect(frame, keypoints);
        
        // Convert KeyPoints to Points2f
        points.clear();
        if (keypoints.empty()) {
            std::cout << "Warning: No features detected!" << std::endl;
            return;
        }
        
        for(const auto& kp : keypoints) {
            points.push_back(kp.pt);
        }
        
        // If we have too many points, keep only the strongest ones
        if (points.size() > static_cast<size_t>(MIN_FEATURES)) {
            points.resize(MIN_FEATURES);
        }
        
        std::cout << "Detected " << points.size() << " features" << std::endl;
    }

    bool trackFeatures() {
        if (prev_points.empty()) {
            std::cout << "No previous points to track" << std::endl;
            return false;
        }

        std::vector<uchar> status;
        std::vector<float> err;
        
        curr_points.clear();  // Ensure curr_points is empty
        
        cv::calcOpticalFlowPyrLK(prev_frame, curr_frame, prev_points, curr_points, 
                                status, err, cv::Size(21, 21), 3);

        // Filter out bad matches
        std::vector<cv::Point2f> good_prev, good_curr;
        for(size_t i = 0; i < status.size(); i++) {
            if(status[i]) {
                good_prev.push_back(prev_points[i]);
                good_curr.push_back(curr_points[i]);
            }
        }
        
        prev_points = good_prev;
        curr_points = good_curr;
        
        std::cout << "Tracked " << prev_points.size() << " points" << std::endl;
        
        return prev_points.size() >= 8; // Minimum points needed for Essential Matrix
    }

    void estimateMotion() {
        if (curr_points.size() < 8 || prev_points.size() < 8) {
            std::cout << "Not enough points for motion estimation" << std::endl;
            return;
        }

        // Calculate Essential Matrix
        cv::Mat E = cv::findEssentialMat(curr_points, prev_points, K, 
                                        cv::RANSAC, 0.999, 1.0);
        
        // Recover R and t from Essential Matrix
        cv::Mat R, t;
        cv::recoverPose(E, curr_points, prev_points, K, R, t);
        
        // Update global pose
        t_f = t_f + R_f * t;
        R_f = R * R_f;
    }

    void processFrame(const cv::Mat& frame) {
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        
        if (!is_initialized) {
            prev_frame = gray.clone();
            detectFeatures(prev_frame, prev_points);
            is_initialized = true;
            std::cout << "Initialized with " << prev_points.size() << " features" << std::endl;
            return;
        }
        
        curr_frame = gray.clone();
        
        if (trackFeatures()) {
            estimateMotion();
            
            // If we lost too many features, detect new ones
            if(prev_points.size() < static_cast<size_t>(MIN_FEATURES)) {
                std::cout << "Detecting new features..." << std::endl;
                detectFeatures(curr_frame, prev_points);
            }
        } else {
            // If tracking failed, reinitialize
            std::cout << "Tracking failed, reinitializing..." << std::endl;
            detectFeatures(curr_frame, prev_points);
        }
        
        prev_frame = curr_frame.clone();
    }

    cv::Mat getCurrentRotation() const { return R_f; }
    cv::Mat getCurrentTranslation() const { return t_f; }

    void drawFeatures(cv::Mat& frame) {
        for(const auto& pt : prev_points) {
            cv::circle(frame, pt, 3, cv::Scalar(0, 255, 0), -1);
        }
        
        // Draw status text
        std::string status = "Features: " + std::to_string(prev_points.size());
        cv::putText(frame, status, cv::Point(10, 30), 
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
    }
};

int main() {
    // Camera matrix (example values - adjust based on your camera calibration)
    cv::Mat K = (cv::Mat_<double>(3,3) << 
        718.8560, 0.0, 607.1928,
        0.0, 718.8560, 185.2157,
        0.0, 0.0, 1.0);
    
    VisualOdometry vo(K);
    cv::VideoCapture cap(0); // Use camera index 0
    
    if(!cap.isOpened()) {
        std::cout << "Error opening camera" << std::endl;
        return -1;
    }
    
    cv::Mat frame;
    while(cap.read(frame)) {
        cv::Mat display = frame.clone();
        
        vo.processFrame(frame);
        vo.drawFeatures(display);
        
        // Get current position
        cv::Mat R = vo.getCurrentRotation();
        cv::Mat t = vo.getCurrentTranslation();
        
        // Display trajectory or current position
        std::cout << "Translation: " << t.t() << std::endl;
        
        // Show frame with features
        cv::imshow("Visual Odometry", display);
        
        if(cv::waitKey(1) == 27) // ESC key
            break;
    }
    
    return 0;
}