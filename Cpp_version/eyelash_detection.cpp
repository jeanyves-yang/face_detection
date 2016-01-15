#include <stdio.h>
#include <opencv2/opencv.hpp>

void eyelash_detection(cv::Mat src, cv::Rect rectEye, cv::Mat &vis, cv::vector<cv::Point> &eyeLash)
{

    cv::Point BR=rectEye.br();
    cv::Point TL=rectEye.tl();
    cv::Mat eyeROI = src( rectEye );
    cv::Mat hsvEyeROI;
    cv::cvtColor(eyeROI, hsvEyeROI, CV_RGB2HSV);

    cv::Mat splitHsvEyeROI[3];
    cv::split(hsvEyeROI,splitHsvEyeROI);
    cv::Mat veyeROI = splitHsvEyeROI[2];
    veyeROI=cv::Mat::ones(veyeROI.size(),CV_8U)*255-veyeROI;

    cv::Mat sob=veyeROI.clone();
    //cv::Sobel(sob,sob,-1,0,1);
    cv::equalizeHist(sob,sob);
    cv::threshold(sob,sob,220,255,CV_THRESH_BINARY);

    cv::vector<cv::vector<cv::Point>> contours;
    cv::findContours(sob,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
    int bigContour1=0;
    int indbigContour1=0;
    int bigContour2=0;
    int indbigContour2=0;

    //get the biggest contour, hopefuly the eye
    for(int i=0;i<contours.size();i++)
    {
        if(contours[i].size()>bigContour1)
        {
            bigContour2 = bigContour1;
            indbigContour2 = indbigContour1;
            bigContour1=contours[i].size();
            indbigContour1=i;
        }
    }

    if (contours[indbigContour2][0].y>contours[indbigContour1][0].y && contours[indbigContour2].size()>30)
    {
        indbigContour1=indbigContour2;
    }


    sob=cv::Mat::zeros(sob.size(),CV_8U);
    cv::drawContours(sob,contours,indbigContour1,(255,255,255),-1);

    //Find extreme points of the contours (to draw the line)
    cv::Point ptLeft;
    ptLeft.x=sob.size().width-1;
    ptLeft.y=0;
    cv::Point ptRight;
    ptRight.x=0;
    ptRight.y=0;

    cv::vector<cv::Point> eyeContour=contours[indbigContour1];
    for(int i=0;i<eyeContour.size();i++)
    {
        if(eyeContour.at(i).x<ptLeft.x)
            ptLeft=eyeContour[i];
        if(eyeContour.at(i).x>ptRight.x)
            ptRight=eyeContour[i];
    }
    ptLeft=ptLeft+TL;
    ptRight=ptRight+TL;

    cv::line(vis,ptLeft,ptRight,cv::Scalar(255,255,0),2);

    eyeLash.empty();
    eyeLash.push_back(ptLeft);
    eyeLash.push_back(ptRight);

}
