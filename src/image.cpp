#include "image.h"

#include <stb_image_write.h>

#include <iostream>
#include <string>
#include <cmath>

namespace
{
    // Convert linear radiance to the sRGB transfer curve for 8-bit display
    // output. Values outside display range are clipped before conversion.
    float linearToSrgb(float value)
    {
        value = std::fmax(0.0f, std::fmin(1.0f, value));
        return value <= 0.0031308f
            ? 12.92f * value
            : 1.055f * std::pow(value, 1.0f / 2.4f) - 0.055f;
    }
}

Image::Image(int x, int y)
    : xSize(x), ySize(y), pixels(new glm::vec3[x * y]) 
{}

Image::~Image()
{
    delete[] pixels;
}

void Image::setPixel(int x, int y, const glm::vec3 &pixel)
{
    assert(x >= 0 && y >= 0 && x < xSize && y < ySize);
    pixels[(y * xSize) + x] = pixel;
}

void Image::savePNG(const std::string &baseFilename)
{
    unsigned char *bytes = new unsigned char[3 * xSize * ySize];
    for (int y = 0; y < ySize; y++)
    {
        for (int x = 0; x < xSize; x++)
        {
            int i = y * xSize + x;
            glm::vec3 pix(
                linearToSrgb(pixels[i].x),
                linearToSrgb(pixels[i].y),
                linearToSrgb(pixels[i].z));
            bytes[3 * i + 0] = (unsigned char) std::lround(pix.x * 255.0f);
            bytes[3 * i + 1] = (unsigned char) std::lround(pix.y * 255.0f);
            bytes[3 * i + 2] = (unsigned char) std::lround(pix.z * 255.0f);
        }
    }

    std::string filename = baseFilename + ".png";
    stbi_write_png(filename.c_str(), xSize, ySize, 3, bytes, xSize * 3);
    std::cout << "Saved " << filename << "." << std::endl;

    delete[] bytes;
}

void Image::saveHDR(const std::string &baseFilename)
{
    std::string filename = baseFilename + ".hdr";
    stbi_write_hdr(filename.c_str(), xSize, ySize, 3, (const float *) pixels);
    std::cout << "Saved " + filename + "." << std::endl;
}
