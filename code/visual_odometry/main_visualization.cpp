#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <deque>

class VisualOdometry {
private:
    cv::Mat K;  // Camera intrinsic matrix
    cv::Mat prev_frame;
    cv::Mat curr_frame;
    std::vector<cv::Point2f> prev_points;
    std::vector<cv::Point2f> curr_points;
    cv::Mat R_f, t_f; // Final rotation and translation
    bool is_initialized;
    
    // Trajectory storage
    std::deque<cv::Point3d> trajectory;
    const size_t MAX_TRAJECTORY_LENGTH = 500;
    
    const int MIN_FEATURES = 1500;
    const double MIN_MATCH_DIST = 30.0;
    
    // Visualization parameters
    const int VIZ_WIDTH = 800;
    const int VIZ_HEIGHT = 600;
    const double SCALE = 1.0;

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

        // Update trajectory
        cv::Point3d current_position(t_f.at<double>(0), t_f.at<double>(1), t_f.at<double>(2));
        trajectory.push_back(current_position);
        if (trajectory.size() > MAX_TRAJECTORY_LENGTH) {
            trajectory.pop_front();
        }
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

    void drawTrajectory(cv::Mat& img) {
        // Clear image
        img = cv::Mat::zeros(VIZ_HEIGHT, VIZ_WIDTH, CV_8UC3);
        
        // Center point for visualization
        cv::Point2d center(VIZ_WIDTH/2, VIZ_HEIGHT/2);
        
        // Draw coordinate axes
        cv::arrowedLine(img, center, 
                        cv::Point2d(center.x + 100, center.y), 
                        cv::Scalar(0,0,255), 2); // X-axis (Red)
        cv::arrowedLine(img, center, 
                        cv::Point2d(center.x, center.y - 100), 
                        cv::Scalar(0,255,0), 2); // Y-axis (Green)
        
        // Draw trajectory
        if (trajectory.size() > 1) {
            for (size_t i = 1; i < trajectory.size(); i++) {
                cv::Point2d pt1(
                    center.x + trajectory[i-1].x * SCALE,
                    center.y - trajectory[i-1].z * SCALE
                );
                cv::Point2d pt2(
                    center.x + trajectory[i].x * SCALE,
                    center.y - trajectory[i].z * SCALE
                );
                cv::line(img, pt1, pt2, cv::Scalar(255,255,0), 2);
            }
        }
        
        // Draw current position
        if (!trajectory.empty()) {
            cv::Point2d current(
                center.x + trajectory.back().x * SCALE,
                center.y - trajectory.back().z * SCALE
            );
            cv::circle(img, current, 5, cv::Scalar(0,255,255), -1);
        }
        
        // Draw status text
        std::string status = "Position - X: " + std::to_string(t_f.at<double>(0)) +
                           " Y: " + std::to_string(t_f.at<double>(1)) +
                           " Z: " + std::to_string(t_f.at<double>(2));
        cv::putText(img, status, cv::Point(10, 30), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255), 1);
    }

    void drawFeatures(cv::Mat& frame) {
        for(const auto& pt : prev_points) {
            cv::circle(frame, pt, 3, cv::Scalar(0, 255, 0), -1);
        }
        
        // Draw status text
        std::string status = "Features: " + std::to_string(prev_points.size());
        cv::putText(frame, status, cv::Point(10, 30), 
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
    }

    cv::Mat getCurrentRotation() const { return R_f; }
    cv::Mat getCurrentTranslation() const { return t_f; }
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
    
    cv::Mat frame, trajectory_view;
    
    // Create windows
    cv::namedWindow("Visual Odometry", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Trajectory", cv::WINDOW_AUTOSIZE);
    
    while(cap.read(frame)) {
        cv::Mat display = frame.clone();
        
        vo.processFrame(frame);
        vo.drawFeatures(display);
        
        // Create and update trajectory visualization
        trajectory_view = cv::Mat::zeros(600, 800, CV_8UC3);
        vo.drawTrajectory(trajectory_view);
        
        // Show both windows
        cv::imshow("Visual Odometry", display);
        cv::imshow("Trajectory", trajectory_view);
        
        if(cv::waitKey(1) == 27) // ESC key
            break;
    }
    
    return 0;
}