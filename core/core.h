#pragma once  
#include <opencv2/opencv.hpp>  
#include <string>  

namespace core  
{  
	/**  
	 * @brief Test if OpenCV is properly configured and working.  
	 */  
	void TestOpencv();  

	/**  
	 * @brief Perform Harris Corner Detection on an input image.  
	 * @param image Input image (grayscale or color).  
	 * @param blockSize Neighborhood size for corner detection.  
	 * @param ksize Aperture parameter for the Sobel operator.  
	 * @param k Harris detector free parameter.  
	 * @return A matrix containing the Harris response.  
	 */  
	cv::Mat HarrisCornerDetection(const cv::Mat& image, int blockSize, int ksize, double k);  

	/**  
	 * @brief Print a standard message to the console.  
	 * @param message The message to print.  
	 */  
	void PrintMessage(const std::string& message);  

	/**  
	 * @brief Print an error message to the console.  
	 * @param message The error message to print.  
	 */  
	void PrintErrorMessage(const std::string& message);  

	/**  
	 * @brief Handle a command with its arguments.  
	 * @param cmd The command to handle.  
	 * @param args The arguments for the command.  
	 */  
	void Handle(const std::string& cmd, char** args);  

	/**  
	 * @brief Compute Sobel gradients (Ix, Iy) with a kernel size of 5.  
	 * @param image Input image (grayscale or color).  
	 * @param Ix Output gradient in the x-direction.  
	 * @param Iy Output gradient in the y-direction.  
	 */  
	void ComputeSobelGradientsSize5(const cv::Mat& image, cv::Mat& Ix, cv::Mat& Iy);  

	/**  
	 * @brief Convert an RGB image to grayscale.  
	 * @param src Input RGB image.  
	 * @return Grayscale image.  
	 */  
	cv::Mat RGBToGray(const cv::Mat& src);  

	/**  
	 * @brief Generate a Gaussian window of a given size.  
	 * @param size Size of the Gaussian window.  
	 * @return A matrix representing the Gaussian window.  
	 */  
	cv::Mat GaussianWindow(int size);  

	/**  
	 * @brief Apply a Gaussian filter to an image.  
	 * @param src Input image.  
	 * @param kernel Size of the Gaussian kernel (default is 3).  
	 * @return Filtered image.  
	 */  
	cv::Mat GaussianFilter(const cv::Mat& src, int kernel = 3);  

	/**  
	 * @brief Display a 32-bit single-channel image with a given name.  
	 * @param src Input image.  
	 * @param name Window name for display.  
	 * @return The input image for further processing.  
	 */  
	cv::Mat Display32FC1(const cv::Mat& src, const std::string& name);  

	/**  
	 * @brief Visualize Harris corners on an image.  
	 * @param image Input image (modified in-place).  
	 * @param harrisResponse Harris response matrix.  
	 */  
	void VisualizeHarrisCorners(cv::Mat& image, const cv::Mat& harrisResponse);  

	/**  
	 * @brief Compute the Harris response matrix from image gradients.  
	 * @param Ix Gradient in the x-direction.  
	 * @param Iy Gradient in the y-direction.  
	 * @param blockSize Neighborhood size for corner detection.  
	 * @param k Harris detector free parameter.  
	 * @return Harris response matrix.  
	 */  
	cv::Mat ComputeHarrisResponse(const cv::Mat& Ix, const cv::Mat& Iy, int blockSize, double k);  

	/**  
	 * @brief Apply a threshold to an image.  
	 * @param image Input image (modified in-place).  
	 * @param threshold Threshold value (default is 1e9).  
	 */  
	void ApplyThreshold(cv::Mat& image, float threshold = 1000000000.f);  

	/**  
	 * @brief Perform non-maximum suppression on an image.  
	 * @param image Input image (modified in-place).  
	 * @param windowSize Size of the suppression window (default is 5).  
	 */  
	void NonMaxSuppression(cv::Mat& image, int windowSize = 5);  
}