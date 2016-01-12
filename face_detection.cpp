#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;

int main(int argc, char** argv )
{
    if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }

    Mat image;
    image = imread( argv[1], 1 );

    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }

    int rows = image.rows;
    int cols = image.cols;

    // Adapt input image resolution to fit Viola Jones training done by OpenCV
    resize(image, image, Size(), 0.05, 0.05, INTER_LINEAR);
    transpose(image,image);
    flip(image,image,cols/2);

    // Load Cascade Classifiers
    CascadeClassifier faceCascade;
    faceCascade.load( "/fs03/share/users/jean-yves.yang/home/face_detection/haarcascade_frontalface_alt.xml" );

    CascadeClassifier eyeCascade;
    eyeCascade.load( "/fs03/share/users/jean-yves.yang/home/face_detection/haarcascade_eye.xml" );

    CascadeClassifier mouthCascade;
    mouthCascade.load( "/fs03/share/users/jean-yves.yang/home/face_detection/haarcascade_mcs_mouth.xml" );

    Mat gray;

    cvtColor(image, gray, CV_BGR2GRAY);

    // Apply 2D Convolution
    Mat kernel;
    int kernelSize = 5;
    kernel = Mat::ones( kernelSize, kernelSize, CV_32F )/ (float)(kernelSize * kernelSize);
    filter2D(gray, gray, -1 , kernel, Point( 0, 0 ), 0, BORDER_DEFAULT );

    // Detect Face
    std::vector<Rect> rectFace;
    faceCascade.detectMultiScale(gray, rectFace,1.1, 3,0|CV_HAAR_SCALE_IMAGE,Size(10,10));

    Mat faceROI = gray( rectFace[0] );

    // Detect Eyes
    std::vector<Rect> rectEyes;

    eyeCascade.detectMultiScale(faceROI, rectEyes,1.1, 3,0|CV_HAAR_SCALE_IMAGE,Size(10,10));

    // Detect Mouth
    std::vector<Rect> rectMouth;
    mouthCascade.detectMultiScale(faceROI, rectMouth,1.1, 3,0|CV_HAAR_SCALE_IMAGE,Size(10,10));

    Mat vis;
    image.copyTo(vis);
    rectangle(vis, rectFace[0], Scalar(255,0,0),2);

    for( size_t j = 0; j < rectEyes.size(); j++ )
    {
        Point TL = rectEyes[j].tl() + rectFace[0].tl();
        Point BR = rectEyes[j].br() + rectFace[0].tl() ;
        rectangle(vis, TL, BR, Scalar(0,255,0), 2);
    }

    Point TL = rectMouth[0].tl() + rectFace[0].tl();
    Point BR = rectMouth[0].br() + rectFace[0].tl();

    rectangle(vis, TL, BR, Scalar(0,0,255),2);

    namedWindow("Display Image", WINDOW_NORMAL);
    imshow("Display Image", vis);

    waitKey(0);
    return 0;
}
