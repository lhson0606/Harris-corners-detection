#include <opencv2/opencv.hpp>
#include "core/core.h"

int main(int argc, char** args)
{
	if (argc != 4)
	{
		core::PrintErrorMessage("Usage: <program> -harris <image_path> <output_path>");
	}

	core::Handle(args[1], args);

	return 0;
}