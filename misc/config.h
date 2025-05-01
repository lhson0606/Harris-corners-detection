#pragma once
#include <string>

namespace config
{
	static const std::string kHarris = "-harris";
	static const std::string INPUT_DIR = "Data/input/";
	static const std::string OUTPUT_DIR = "Data/output/";

	// Pixel values
	static const int MAX_PIXEL_VALUE = 255;
	static const int MIN_PIXEL_VALUE = 0;

	static const int RED_INDEX = 2;
	static const int GREEN_INDEX = 1;
	static const int BLUE_INDEX = 0;
}