#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>

//from the original image, return the rects which square each ROI
void face_detection(cv::Mat& ,cv::Rect& , cv::vector<cv::Rect>& ,cv::Rect& ,cv::Mat&);

//Detect the eyelash on the image, the rect square the eyeROI and the fct "return" a vect of 2 points which is the line of the eyelash
void eyelash_detection(cv::Mat, cv::Rect, cv::Mat&, cv::vector<cv::Point>&);

//Detect the lips on the image, the rect square the mouthROI and the fct "return" a vect of 2 points which is the line of the lips
void lips_detection(cv::Mat, cv::Rect,cv::Rect, cv::Mat&, cv::vector<cv::Point>&);

//write in a file .txt the detection realized on this image
//Read the file of the previous image to give the comparison
void tracking(const std::string&, const cv::vector<cv::vector<cv::Point>>);
