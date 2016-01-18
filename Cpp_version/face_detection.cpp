#include <stdio.h>
#include <opencv2/opencv.hpp>

void face_detection(cv::Mat &image,cv::Rect &rectFace, cv::vector<cv::Rect> &rectEyes, cv::Rect &rectMouth ,cv::Mat &vis)
{
    int cols = image.cols;

    // Adapt input image resolution to fit Viola Jones training done by OpenCV
    cv::resize(image, image, cv::Size(), 0.10, 0.10, cv::INTER_LINEAR);
    cv::transpose(image,image);
    cv::flip(image,image,cols/2);

    vis=image.clone();

    // Load Cascade Classifiers
    cv::CascadeClassifier faceCascade;
    faceCascade.load( "../Cpp_version/haarcascade_frontalface_alt.xml" );

    cv::CascadeClassifier eyeCascade;
    eyeCascade.load( "../Cpp_version/haarcascade_eye.xml" );

    cv::CascadeClassifier mouthCascade;
    mouthCascade.load( "../Cpp_version/haarcascade_mcs_mouth.xml" );

    cv::Mat gray;

    cvtColor(image, gray, CV_BGR2GRAY);

    // Apply 2D Convolution
    cv::Mat kernel;
    int kernelSize = 5;
    kernel = cv::Mat::ones( kernelSize, kernelSize, CV_32F )/ (float)(kernelSize * kernelSize);
    cv::filter2D(gray, gray, -1 , kernel, cv::Point( 0, 0 ), 0, cv::BORDER_DEFAULT );

    //detect face
    cv::Mat srcLowScale = gray.clone();
    cv::resize(srcLowScale, srcLowScale, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);
    cv::vector<cv::Rect> rectFaces;
    faceCascade.detectMultiScale( srcLowScale, rectFaces,1.1, 3,0|CV_HAAR_SCALE_IMAGE,cv::Size(10,10));
    rectFace = cv::Rect(rectFaces[0].tl()*2, rectFaces[0].br()*2);
    cv::rectangle(vis, rectFace, cv::Scalar(255,0,0),2);

    cv::Mat faceROILowScale = gray(rectFace);
    cv::resize(faceROILowScale, faceROILowScale, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);

    //detect eyes
    cv::vector<cv::Rect> rectEyesUnscaled;
    eyeCascade.detectMultiScale( faceROILowScale, rectEyesUnscaled,1.1, 3,0|CV_HAAR_SCALE_IMAGE,cv::Size(10,10));
    rectEyes.clear();
    rectEyes.push_back(cv::Rect(rectEyesUnscaled[0].tl()*2+rectFace.tl(),rectEyesUnscaled[0].br()*2+rectFace.tl()));
    rectEyes.push_back(cv::Rect(rectEyesUnscaled[1].tl()*2+rectFace.tl(),rectEyesUnscaled[1].br()*2+rectFace.tl()));
    for( size_t j = 0; j < rectEyes.size(); j++ )
    {
        cv::rectangle(vis, rectEyes[j], cv::Scalar(0,255,0),2);
    }

    // Detect Mouth
    std::vector<cv::Rect> rectMouths;
    mouthCascade.detectMultiScale(faceROILowScale, rectMouths,1.1, 3,0|CV_HAAR_SCALE_IMAGE,cv::Size(10,10));
    //Take the mouth the most at the bottom
    int indGoodMouth=0;
    int posGoodMouth=0;
    for( size_t j = 0; j < rectMouths.size(); j++ )
    {
        if(rectMouths[j].br().y>posGoodMouth)
        {
            indGoodMouth=j;
            posGoodMouth=rectMouths[j].br().y;
        }
    }

    rectMouth = cv::Rect(rectMouths[indGoodMouth].tl()*2+rectFace.tl(),rectMouths[indGoodMouth].br()*2+rectFace.tl());
    cv::rectangle(vis, rectMouth, cv::Scalar(0,0,255),2);

}
