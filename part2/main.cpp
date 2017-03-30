////
////  main.cpp
////  ProgramAssignment1
////
////  Created by 周思雨 on 9/30/16.
////  Copyright © 2016 周思雨. All rights reserved.
////



#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <opencv/cv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
//#include <tr1/unordered_map>


using namespace cv;
using namespace std;

Mat hisEqualization(Mat source, string savename, int iw, int ih){
    savename = "/Users/Jocy/Dropbox/2016 Fall/CSI 111/Program assignment/ProgramAssignment4/part2/";
    
    int histSize = 256; // bin size
    float range[] = { 0, 256} ;
    const float* histRange = { range };
    bool uniform = true;
    bool accumulate = false;
    Mat hist;
    int channels[] = {0};
    /// Compute the histograms:
    calcHist( &source, 1, channels, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate );
    //cout<<hist<<endl;
    Mat cdf = hist;
    cdf.at<float>(0) = hist.at<float>(0);
    for(int i = 1; i < 256; i++){
        cdf.at<float>(i) = cdf.at<float>(i - 1) + hist.at<float>(i);
    }
    
    
    cv::Mat dest = Mat(iw,ih, CV_8UC1);
    double min, max;
    cv::minMaxLoc(hist, &min, &max);
    for(int i = 0; i < ih; i++){
        for(int j = 0; j < iw; j++){
            dest.at<uchar>(i, j) = round(255*(cdf.at<float>(source.at<uchar>(i, j)) - min)/((iw * ih) - min));
            //cout<<dest.at<uchar>(i, j)<<endl;
        }
    }
    imshow("equalization", dest);
    cvWaitKey(0);
    savename = savename + "ucihisEqualization.jpg";
    imwrite(savename, dest);
    return dest;
}


int main(int argc, char* argv[]) {
    // insert code here...
    std::cout << "Hello, World!\n";
    cv::Mat img = cv::imread(argv[1], 1);
    std::string savename = "/Users/Jocy/Dropbox/2016 Fall/CSI 111/Program assignment/ProgramAssignment2/flowergray/part1/flowergray";
    if (img.empty()) {
        std::cout << "Error : Image loading failed!" << std::endl;
        return -1;
    }
    cv::Mat source = cvCreateMat(img.rows, img.cols, CV_8UC1);
    if (img.channels() != 1){
        std::cout << "Warning : Input image is not a grayscale image" << std::endl;
        cv::cvtColor(img, source, CV_RGBA2GRAY);
    }else{
        source = img;
    }
    //float gaussian[3][3] = {{-0.125, -0.125, -0.125}, {-0.125, 1, -0.125}, {-0.125, -0.125, -0.125}};
    float gaussian[5] = {0.05, 0.25, 0.4, 0.25, 0.05};
    cv::Mat filter = Mat(1,5, CV_32F, gaussian);
    
//    if(source.cols != 256 || source.rows != 256){
//        source = imresize(source, source.cols, source.rows, 256, 256);
//    }
    
    //PA4-part3
    //medFilter(source, savename, source.cols, source.rows, 10, 10);
    
    //PA4-part2
    cv::Mat dst = hisEqualization(source, savename, source.cols, source.rows);
    
    //PA4-part1
    //secDerivative(source, savename, filter, source.cols, source.rows, filter.cols, filter.rows);
    
    
    //cv::Mat filtertri = Mat(3, 3, CV_32F, Scalar(1.0/9));
    //imresize(source, source.cols, source.rows, (source.cols) * 0.75, (source.rows) * 0.75);
    
    //cout << filtertri << endl;
    //pyramida(source,savename, filter,source.cols, source.rows, filter.cols, filter.rows);
    //pyramidb(source,savename, filter,source.cols, source.rows, filter.cols, filter.rows);
    //convolve(source, filter,source.cols, source.rows, filter.cols, filter.rows);
    //convolve(source, filtertri, source.cols, source.rows, filtertri.cols, filtertri.rows);
    
    //Lappyramid(source, savename, filter, source.cols, source.rows, filter.cols, filter.rows);
    //Highpassa(source, savename, filter, source.cols, source.rows, filter.cols, filter.rows);
    //Highpassb(source, savename, filter, source.cols, source.rows, filter.cols, filter.rows);
    return -1;
}
