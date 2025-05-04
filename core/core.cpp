#include "core.h"
#include <iostream>
#include "../misc/config.h"
#include "../misc/utils.h"
#include <exception>

void core::TestOpencv()
{
	// Test OpenCV functionality
	cv::Mat image = cv::imread("test.jpg");
	if (image.empty())
	{
		std::cerr << "Could not open or find the image!" << std::endl;
		return;
	}
	// Display the image in a window
	cv::imshow("Test Image", image);
	cv::waitKey(0); // Wait for a key press indefinitely
}

cv::Mat core::HarrisCornerDetection(const cv::Mat& image, int blockSize, int ksize, double k)
{
	// ref (implementation): https://docs.nvidia.com/vpi/algo_harris_corners.html
	// ref (explanation): https://www.youtube.com/watch?v=Z_HwkG90Yvw

	// Step 1: Convert to grayscale
	cv::Mat gray = RGBToGray(image);

	// Step 2: Compute spatial derivatives using Sobel operator
	cv::Mat Ix, Iy;
	ComputeSobelGradientsSize5(gray, Ix, Iy);

	// Step 3: Compute Harris response
	cv::Mat harrisResponse = ComputeHarrisResponse(Ix, Iy, blockSize, k);

	// Step 4: Thresholding
	cv::Mat harrisVisual = image.clone();
	ApplyThreshold(harrisResponse);

	// Step 5: Non-max supppression
	NonMaxSuppression(harrisResponse);

	// Display32FC1(harrisResponse, "Harris response");

	VisualizeHarrisCorners(harrisVisual, harrisResponse);
	
	return harrisVisual;
}

cv::Mat core::ComputeHarrisResponse(const cv::Mat& Ix, const cv::Mat& Iy, int blockSize, double k)
{
	cv::Mat harrisResponse = cv::Mat(Ix.size(), CV_32FC1);
	int halfWindowSize = blockSize / 2;

	cv::Mat smoothedIx = GaussianFilter(Ix, blockSize);
	cv::Mat smoothedIy = GaussianFilter(Iy, blockSize);

	for (int y = 0; y < Ix.rows; y++)
	{
		for (int x = 0; x < Ix.cols; x++)
		{
			if (x < halfWindowSize || y < halfWindowSize || x >= Ix.cols - halfWindowSize || y >= Ix.rows - halfWindowSize)
			{
				harrisResponse.at<float>(y, x) = 0;
				continue;
			}

			float Ixx = 0.0f, Iyy = 0.0f, Ixy = 0.0f;
			for (int i = -halfWindowSize; i <= halfWindowSize; i++)
			{
				for (int j = -halfWindowSize; j <= halfWindowSize; j++)
				{
					float ix = smoothedIx.at<float>(y + i, x + j);
					float iy = smoothedIy.at<float>(y + i, x + j);
					Ixx += ix * ix;
					Iyy += iy * iy;
					Ixy += ix * iy;
				}
			}

			float detM = Ixx * Iyy - Ixy * Ixy;
			float traceM = Ixx + Iyy;
			harrisResponse.at<float>(y, x) = detM - k * traceM * traceM;
		}
	}

	return harrisResponse;
}

void core::ApplyThreshold(cv::Mat& image, float threshold)
{
	for (int y = 0; y < image.rows; y++)
	{
		for (int x = 0; x < image.cols; x++)
		{
			if (image.at<float>(y, x) < threshold)
			{
				image.at<float>(y, x) = 0;
			}
		}
	}
}

void core::NonMaxSuppression(cv::Mat& image, int windowSize)
{
	auto res = image.clone();
	int halfWindowSize = windowSize / 2;
	for (int y = halfWindowSize; y < image.rows - halfWindowSize; y++)
	{
		for (int x = halfWindowSize; x < image.cols - halfWindowSize; x++)
		{
			float center = image.at<float>(y, x);
			bool isMax = true;

			for (int i = -halfWindowSize; i <= halfWindowSize && isMax; i++)
			{
				for (int j = -halfWindowSize; j <= halfWindowSize; j++)
				{
					if (i == 0 && j == 0)
						continue;
					float neighbor = image.at<float>(y + i, x + j);
					if (neighbor > center)
					{
						isMax = false;
						break;
					}
				}
			}

			res.at<float>(y, x) = isMax ? center : 0;
		}
	}

	image = res.clone();
}

void core::VisualizeHarrisCorners(cv::Mat& image, const cv::Mat& harrisResponse)
{
	for (int y = 0; y < harrisResponse.rows; y++)
	{
		for (int x = 0; x < harrisResponse.cols; x++)
		{
			if (harrisResponse.at<float>(y, x) > 0)
			{
				cv::circle(image, cv::Point(x, y), 1, cv::Scalar(0, 255, 0), -1);
			}
		}
	}
}

void core::PrintMessage(const std::string& message)
{
	std::cout << message << std::endl;
}

void core::PrintErrorMessage(const std::string& message)
{
	std::cerr << "Error: " << message << std::endl;
}

void core::Handle(const std::string& cmd, char** args)
{
	if (cmd == config::kHarris)
	{
		if (args[2] == nullptr || args[3] == nullptr)
		{
			PrintErrorMessage("Invalid arguments for -harris");
			return;
		}
		const std::string imagePath = config::INPUT_DIR + args[2];
		cv::Mat image = cv::imread(imagePath);
		// cv::imshow("Harris Corner Detection", image);
		if (image.empty())
		{
			PrintErrorMessage("Could not open or find the image!");
			return;
		}
		cv::Mat res = HarrisCornerDetection(image, 5, 5, 0.04);
		const std::string savePath = config::OUTPUT_DIR + args[3];
		cv::imwrite(savePath, res);
		PrintMessage("Harris corner detection completed. Result saved to " + savePath);
		cv::imshow("Harris Corner Detection", res);
		cv::waitKey(0); // Wait for a key press indefinitely
	}
	else
	{
		PrintErrorMessage("Unknown command");
	}
}

void core::ComputeSobelGradientsSize5(const cv::Mat& image, cv::Mat& Ix, cv::Mat& Iy)
{
	// Input image must be grayscale
	if (image.channels() != 1)
	{
		throw std::runtime_error("Input image must be grayscale");
	}

	Ix = cv::Mat(image.size(), CV_32FC1);
	Iy = cv::Mat(image.size(), CV_32FC1);

	cv::Mat sobelX = (cv::Mat_<float>(5, 5) <<
		-1, -2, 0, 2, 1,
		-4, -8, 0, 8, 4,
		-6, -12, 0, 12, 6,
		-4, -8, 0, 8, 4,
		-1, -2, 0, 2, 1) / 48.0;

	cv::Mat sobelY = (cv::Mat_<float>(5, 5) <<
		-1, -4, -6, -4, -1,
		-2, -8, -12, -8, -2,
		0, 0, 0, 0, 0,
		2, 8, 12, 8, 2,
		1, 4, 6, 4, 1) / 48.0;

	for (int y = 0; y < image.rows; y++)
	{
		for (int x = 0; x < image.cols; x++)
		{
			float sumX = 0;
			float sumY = 0;

			// Handle boundary conditions
			if (x < 2 || y < 2 || x >= image.cols - 2 || y >= image.rows - 2)
			{
				Ix.at<float>(y, x) = 0;
				Iy.at<float>(y, x) = 0;
				continue;
			}

			// Apply Sobel kernels
			for (int i = -2; i <= 2; i++)
			{
				for (int j = -2; j <= 2; j++)
				{
					int x1 = x + j;
					int y1 = y + i;
					sumX += image.at<float>(y1, x1) * sobelX.at<float>(i + 2, j + 2);
					sumY += image.at<float>(y1, x1) * sobelY.at<float>(i + 2, j + 2);
				}
			}

			// Assign raw gradient values
			Ix.at<float>(y, x) = sumX;
			Iy.at<float>(y, x) = sumY;
		}
	}
}

cv::Mat core::RGBToGray(const cv::Mat& src)
{
	cv::Mat gray(src.size(), CV_32FC1);

	for (int y = 0; y < src.rows; y++)
	{
		for (int x = 0; x < src.cols; x++)
		{
			cv::Vec3b pixel = src.at<cv::Vec3b>(y, x);
			float grayValue = pixel[config::RED_INDEX] * 0.299 + pixel[config::GREEN_INDEX] * 0.587 + pixel[config::BLUE_INDEX] * 0.114;
			gray.at<float>(y, x) = grayValue;
		}
	}

	return gray;
}

cv::Mat core::GaussianWindow(int size)
{
	cv::Mat window(size, size, CV_32F);
	float sigma = 1.0f;
	float sum = 0.0f;
	for (int y = -size / 2; y <= size / 2; y++)
	{
		for (int x = -size / 2; x <= size / 2; x++)
		{
			float value = utils::gaussian(x, y, sigma);
			window.at<float>(y + size / 2, x + size / 2) = value;
			sum += value;
		}
	}
	window /= sum; // Normalize the window
	return window;
}

cv::Mat core::GaussianFilter(const cv::Mat& src, int kernel)
{
	if (kernel % 2 == 0)
	{
		throw std::invalid_argument("Kernel size must be an odd number");
	}

	cv::Mat res(src.size(), CV_32FC1);
	int k = kernel / 2;

	for (int y = 0; y < src.rows; y++)
	{
		for (int x = 0; x < src.cols; x++)
		{
			float sum = 0.0f;
			float weightSum = 0.0f;
			for (int i = -k; i <= k; i++)
			{
				for (int j = -k; j <= k; j++)
				{
					int x1 = x + j;
					int y1 = y + i;
					if (x1 < 0 || x1 >= src.cols || y1 < 0 || y1 >= src.rows)
					{
						continue;
					}
					float weight = utils::gaussian(j, i);
					sum += src.at<float>(y1, x1) * weight;
					weightSum += weight;
				}
			}
			res.at<float>(y, x) = sum / weightSum;
		}
	}

	return res;
}

cv::Mat core::Display32FC1(const cv::Mat& src, const std::string& name)
{
	cv::Mat display;
	cv::normalize(src, display, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	cv::imshow(name, display);
	return display;
}
