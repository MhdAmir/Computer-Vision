#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>

class CameraCalibrator {
private:
    cv::Size boardSize;                  // Chessboard dimensions
    float squareSize;                    // Size of chessboard square in real world units
    std::vector<std::vector<cv::Point3f>> objectPoints;  // 3D points in real world space
    std::vector<std::vector<cv::Point2f>> imagePoints;   // 2D points in image plane
    cv::Mat cameraMatrix;                // Intrinsic camera matrix
    cv::Mat distCoeffs;                  // Distortion coefficients
    std::vector<cv::Mat> rvecs;          // Rotation vectors
    std::vector<cv::Mat> tvecs;          // Translation vectors
    int imageCount;                      // Counter for captured images
    
    std::vector<cv::Point3f> createObjectPoints() {
        std::vector<cv::Point3f> corners;
        for(int i = 0; i < boardSize.height; i++) {
            for(int j = 0; j < boardSize.width; j++) {
                corners.push_back(cv::Point3f(j * squareSize, i * squareSize, 0.0f));
            }
        }
        return corners;
    }

public:
    CameraCalibrator(cv::Size board_size = cv::Size(9, 6), float square_size = 1.0f) 
        : boardSize(board_size), squareSize(square_size), imageCount(0) {
        cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
        distCoeffs = cv::Mat::zeros(8, 1, CV_64F);
    }

    bool processFrame(const cv::Mat& frame, bool captureImage = false) {
        cv::Mat frameCopy = frame.clone();
        std::vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(frame, boardSize, corners,
            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);

        if (found) {
            cv::Mat gray;
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));

            cv::drawChessboardCorners(frameCopy, boardSize, corners, found);

            if (captureImage) {
                imageCount++;
                std::stringstream ss;
                ss << "calibration_image_" << imageCount << ".jpg";
                cv::imwrite(ss.str(), frame);
                
                imagePoints.push_back(corners);
                objectPoints.push_back(createObjectPoints());
                
                std::cout << "Captured image " << imageCount << std::endl;
            }
        }

        std::string msg = "Captured Images: " + std::to_string(imageCount);
        cv::putText(frameCopy, msg, cv::Point(10, 20), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

        if (found) {
            msg = "Chessboard detected - Press 'a' to capture";
            cv::putText(frameCopy, msg, cv::Point(10, 40), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        }

        cv::imshow("Camera Calibration", frameCopy);
        return found;
    }

    double calibrate(cv::Size imageSize) {
        if (imagePoints.size() < 3) {
            std::cout << "Need at least 3 images for calibration!" << std::endl;
            return -1;
        }
        return cv::calibrateCamera(objectPoints, imagePoints, imageSize,
            cameraMatrix, distCoeffs, rvecs, tvecs,
            cv::CALIB_RATIONAL_MODEL | cv::CALIB_FIX_K4 | cv::CALIB_FIX_K5);
    }

    void saveCalibration(const std::string& filename) {
        cv::FileStorage fs(filename, cv::FileStorage::WRITE);
        fs << "camera_matrix" << cameraMatrix;
        fs << "distortion_coefficients" << distCoeffs;
        fs.release();
    }

    int getImageCount() const { return imageCount; }
    cv::Mat getCameraMatrix() const { return cameraMatrix; }
    cv::Mat getDistCoeffs() const { return distCoeffs; }
};

int main() {
    cv::VideoCapture cap(0); 
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera" << std::endl;
        return -1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

    CameraCalibrator calibrator(cv::Size(9, 6), 25.0); // 9x6 chessboard with 25mm squares
    cv::Size imageSize;
    bool isCalibrated = false;

    std::cout << "Press 'a' to capture image when chessboard is detected" << std::endl;
    std::cout << "Press 'c' to start calibration" << std::endl;
    std::cout << "Press 'ESC' to exit" << std::endl;

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        if (imageSize.empty()) {
            imageSize = frame.size();
        }

        char key = (char)cv::waitKey(1);
        
        bool found = calibrator.processFrame(frame, key == 'a');

        if (key == 'c' && !isCalibrated) {
            std::cout << "Starting calibration..." << std::endl;
            double rms = calibrator.calibrate(imageSize);
            if (rms >= 0) {
                std::cout << "Calibration completed with RMS error: " << rms << std::endl;
                calibrator.saveCalibration("camera_calibration.yml");
                isCalibrated = true;
            }
        }
        else if (key == 27) {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}