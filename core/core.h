#pragma once
#include <opencv2/opencv.hpp>
#include <string>

namespace core
{
	void TestOpencv();
	void HarrisCornerDetection(const cv::Mat& image, int blockSize, int ksize, double k);
	void PrintMessage(const std::string& message);
	void PrintErrorMessage(const std::string& message);
	void Handle(const std::string& cmd, char** args);
	void ComputeSobelGradientsSize5(const cv::Mat& image, cv::Mat& Ix, cv::Mat& Iy);
	cv::Mat RGBToGray(const cv::Mat& src);
	cv::Mat GaussianWindow(int size);
	cv::Mat GaussianFilter(const cv::Mat& src, int kernel = 3);
	cv::Mat Display32FC1(const cv::Mat& src, const std::string& name);
	void VisualizeHarrisCorners(cv::Mat& image, const cv::Mat& harrisResponse);
	cv::Mat ComputeHarrisResponse(const cv::Mat& Ix, const cv::Mat& Iy, int blockSize, double k);
	void ApplyThreshold(cv::Mat& image, float threshold = 1000000000.f);
	void NonMaxSuppression(cv::Mat& image, int windowSize = 5);
}