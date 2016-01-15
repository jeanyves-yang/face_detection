#include <stdio.h>
#include <opencv2/opencv.hpp>

void lips_detection(cv::Mat src, cv::Rect rectMouth, cv::Rect rectFace, cv::Mat &vis, cv::vector<cv::Point> &lips)
{

    cv::Mat hsvSrc;
    cv::cvtColor(src, hsvSrc, CV_RGB2HSV);
    cv::Mat hsvFaceROI = hsvSrc(rectFace);
    cv::Mat splitHsvFaceROI[3];
    cv::split(hsvFaceROI,splitHsvFaceROI);

    cv::Mat hsvMouthROI = hsvSrc( rectMouth );
    cv::Mat splitHsvMouthROI[3];
    cv::split(hsvMouthROI,splitHsvMouthROI);
    cv::Mat hMouthROI = splitHsvMouthROI[0];

    cv::Mat hist;
    int histSize = MAX( 25, 2 );
    float hue_range[] = { 0, 180 };
    const float* ranges = { hue_range };

    // Get the Histogram and normalize it
    cv::calcHist( &hMouthROI, 1, 0, cv::Mat(), hist, 1, &histSize, &ranges, true, false );
    cv::normalize( hist, hist, 0, 255, cv::NORM_MINMAX, -1, cv::Mat() );

    // Get Backprojection
    cv::Mat backproj;
    cv::calcBackProject( &hMouthROI, 1, 0, hist, backproj, &ranges, 1, true );

    cv::threshold(backproj,backproj,150,255,CV_THRESH_BINARY_INV);

    cv::vector<cv::vector<cv::Point>> contours;
    cv::findContours(backproj,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
    int bigContour=0;
    int indbigContour=0;

    //get the biggest contour, hopefuly the mouth
    for(int i=0;i<contours.size();i++)
    {
        if(contours[i].size()>bigContour)
        {
            bigContour=contours[i].size();
            indbigContour=i;
        }
    }


    backproj=cv::Mat::zeros(backproj.size(),CV_8U);
    cv::drawContours(backproj,contours,indbigContour,(255,255,255),-1);

    //Find extreme points of the contours (to draw the line)
    cv::Point ptLeft;
    ptLeft.x=backproj.size().width-1;
    ptLeft.y=0;
    cv::Point ptRight;
    ptRight.x=0;
    ptRight.y=0;

    cv::vector<cv::Point> mouthContour=contours[indbigContour];
    for(int i=0;i<mouthContour.size();i++)
    {
        if(mouthContour.at(i).x<ptLeft.x)
            ptLeft=mouthContour[i];
        if(mouthContour.at(i).x>ptRight.x)
            ptRight=mouthContour[i];
    }
    ptLeft=ptLeft+rectMouth.tl();
    ptRight=ptRight+rectMouth.tl();

    //cv::line(vis,ptLeft,ptRight,cv::Scalar(255,0,255),2);

    // Draw the lips contour
    cv::Mat displayContour;
    cv::vector<cv::vector<cv::Point>> hull;
    cv::vector<cv::Point> convexhull;
    cv::convexHull(contours[indbigContour],convexhull);
    hull.push_back(convexhull);
    cv::cvtColor(hsvSrc( rectMouth ), displayContour, CV_HSV2RGB);


    //shift the contour to putu in big image range
    for(int i=0;i<convexhull.size();i++)
    {
        hull[0][i]=hull[0][i]+rectMouth.tl();
    }
    cv::drawContours(vis,hull,0,cv::Scalar(255,0,255),2);

    lips.push_back(ptLeft);
    lips.push_back(ptRight);

}
