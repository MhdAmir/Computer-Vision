#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <vector>
#include <iostream>

struct ReprojectionError {
    ReprojectionError(const cv::Point2f& observed, const cv::Point3f& point3D, 
                     const cv::Mat& K)
        : observed_(observed), point3D_(point3D) {
        fx_ = K.at<double>(0, 0);
        fy_ = K.at<double>(1, 1);
        cx_ = K.at<double>(0, 2);
        cy_ = K.at<double>(1, 2);
    }

    template <typename T>
    bool operator()(const T* const rvec, const T* const tvec, T* residuals) const {
        T point[3];
        point[0] = T(point3D_.x);
        point[1] = T(point3D_.y);
        point[2] = T(point3D_.z);

        T rotated_point[3];
        ceres::AngleAxisRotatePoint(rvec, point, rotated_point);

        rotated_point[0] += tvec[0];
        rotated_point[1] += tvec[1];
        rotated_point[2] += tvec[2];

        T x_proj = fx_ * rotated_point[0] / rotated_point[2] + cx_;
        T y_proj = fy_ * rotated_point[1] / rotated_point[2] + cy_;

        residuals[0] = x_proj - T(observed_.x);
        residuals[1] = y_proj - T(observed_.y);

        return true;
    }

    static ceres::CostFunction* Create(const cv::Point2f& observed,
                                     const cv::Point3f& point3D,
                                     const cv::Mat& K) {
        return new ceres::AutoDiffCostFunction<ReprojectionError, 2, 3, 3>(
            new ReprojectionError(observed, point3D, K));
    }

private:
    cv::Point2f observed_;
    cv::Point3f point3D_;
    double fx_, fy_, cx_, cy_;
};

class BundleAdjustment {
public:
    BundleAdjustment(const cv::Mat& K) : K_(K.clone()) {}

    bool refinePose(const std::vector<cv::Point3f>& points3D,
                   const std::vector<cv::Point2f>& points2D,
                   cv::Mat& rvec,
                   cv::Mat& tvec,
                   int maxIterations = 100) {
        
        if (points3D.size() != points2D.size() || points3D.empty()) {
            std::cerr << "Invalid input points" << std::endl;
            return false;
        }

        double rotation[3] = {rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2)};
        double translation[3] = {tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2)};

        ceres::Problem problem;

        for (size_t i = 0; i < points3D.size(); ++i) {
            ceres::CostFunction* cost_function = ReprojectionError::Create(
                points2D[i], points3D[i], K_);
            
            problem.AddResidualBlock(
                cost_function,
                nullptr,  
                rotation,
                translation);
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.minimizer_progress_to_stdout = true;
        options.max_num_iterations = maxIterations;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        rvec.at<double>(0) = rotation[0];
        rvec.at<double>(1) = rotation[1];
        rvec.at<double>(2) = rotation[2];
        tvec.at<double>(0) = translation[0];
        tvec.at<double>(1) = translation[1];
        tvec.at<double>(2) = translation[2];

        std::cout << summary.BriefReport() << std::endl;
        return summary.IsSolutionUsable();
    }

private:
    cv::Mat K_;
};

void example_usage() {
    cv::Mat K = (cv::Mat_<double>(3,3) << 
        800, 0, 320,
        0, 800, 240,
        0, 0, 1);

    std::vector<cv::Point3f> points3D;
    points3D.push_back(cv::Point3f(0, 0, 0));
    points3D.push_back(cv::Point3f(1, 0, 0));
    points3D.push_back(cv::Point3f(0, 1, 0));

    cv::Mat rvec = (cv::Mat_<double>(3,1) << 0.1, 0.1, 0.1);
    cv::Mat tvec = (cv::Mat_<double>(3,1) << 1.0, 2.0, 10.0);
    
    std::vector<cv::Point2f> points2D;
    cv::projectPoints(points3D, rvec, tvec, K, cv::Mat(), points2D);

    BundleAdjustment ba(K);
    bool success = ba.refinePose(points3D, points2D, rvec, tvec);

    if (success) {
        std::cout << "Optimized rvec: " << rvec.t() << std::endl;
        std::cout << "Optimized tvec: " << tvec.t() << std::endl;
    }
}