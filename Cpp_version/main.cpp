#include "header.hpp"
#include <iostream>
#include <iomanip>
#include <boost/filesystem.hpp>
#include <boost/version.hpp>
#include <ctime>

using namespace cv;

int main(int argc, char** argv )
{
    if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }

    std::clock_t start;
    start=std::clock();

    Mat image;
    image = imread( argv[1], 1 );

    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }

    cv::Rect rectFace;
    cv::vector<cv::Rect> rectEyes;
    cv::Rect rectMouth;
    cv::Mat vis;
    face_detection(image,rectFace,rectEyes,rectMouth,vis);

    cv::vector<cv::Point> eyeLash;
    cv::vector<cv::vector<cv::Point>> eyeLashs;
    for( size_t j = 0; j < rectEyes.size(); j++ )
    {
        eyelash_detection(image,rectEyes[j],vis,eyeLash);
        eyeLashs.push_back(eyeLash);
        eyeLash.clear();
    }

    //Sort the eyeLashes vector to have the right eye on the first index
    if (eyeLashs.at(1).at(0).x < eyeLashs.at(0).at(0).x)
    {
        eyeLashs.at(0).swap(eyeLashs.at(1));
    }


    cv::vector<cv::Point> lips;
    lips_detection(image,rectMouth,rectFace,vis,lips);

    //cv::namedWindow("Display Image", cv::WINDOW_NORMAL);
    //cv::imshow("Display Image", vis);
    boost::filesystem::path outputPath(argv[1]);
    //outputPath.filename().stem().c_str();
    std::string path = "/tmp/NT_0_05/";
    cv::imwrite(path + outputPath.filename().stem().string() + outputPath.filename().extension().string(),vis);

//    tracking(argv[1],eyeLashs);
    std::cout << "Time: "<<(std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << "ms"<<std::endl;
    //waitKey(0);
    return 0;
}
