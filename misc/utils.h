#pragma once

#include <algorithm>
#include <vector>
#include <opencv2/opencv.hpp>

namespace utils
{
	template <class T>
	T clamp(T value, T minValue, T maxValue)
	{
		return std::max(minValue, std::min(maxValue, value));
	}

	template <class T>
	T median(std::vector<T> a)
	{
		std::sort(a.begin(), a.end());
		return a[a.size() / 2];
	}

	float gaussian(int x, int y, float sigma = 1.0f)
	{
		return (1.0 / (2 * 3.14159 * sigma * sigma)) * exp(-(x * x + y * y) / (2 * sigma * sigma));
	}

	cv::Mat MatMultElementWise(const cv::Mat& a, const cv::Mat& b)
	{
		// Element-wise multiplication of two matrices
		if (a.size() != b.size() || a.type() != b.type())
		{
			throw std::runtime_error("Matrices must be of the same size and type for element-wise multiplication");
		}
		cv::Mat result = a.clone();
		for (int y = 0; y < a.rows; y++)
		{
			for (int x = 0; x < a.cols; x++)
			{
				result.at<float>(y, x) = a.at<float>(y, x) * b.at<float>(y, x);
			}
		}
		return result;
	}
}