#include <iostream>
#include <stdio.h>
#include <fstream>
#include <opencv2/opencv.hpp>

void tracking(const std::string& imgname, cv::vector<cv::vector<cv::Point>> eyeLashs)
{
    std::ofstream OUTthisPicture;
    size_t imgIDInd = imgname.find("_T")+2;
    int IDImg = int(imgname[imgIDInd]);             //T0,T2,T3.... "0"=48
    size_t imgNameEnd = imgname.find(".jpg");
    std::string txtName;

    for (size_t i = 0;i<imgNameEnd;i++)//reright the filename getting only the useful info  ex: nicolas_T1_Front Face_No Filters
    {
        txtName.push_back(imgname[i]);
    }


    txtName=txtName+"_detection.txt";
    OUTthisPicture.open (txtName);
    OUTthisPicture << eyeLashs.at(0)<<"\n";
    OUTthisPicture << eyeLashs.at(1)<<"\n";
    OUTthisPicture.close();

    std::ifstream INthisPicture;
    INthisPicture.open (txtName);
    cv::vector<cv::Point> test;
    std::string strLash1,strLash2;
    std::getline(INthisPicture,strLash1);     // [250, 230; 297, 244]
    std::getline(INthisPicture,strLash2);

    std::cout<<strLash1<<std::endl;
    std::cout<<strLash2<<std::endl;





}
//nicolas_T1_Front Face_No Filters.jpg
