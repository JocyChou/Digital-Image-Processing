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

Mat imresize(Mat source, int iw, int ih, int tw, int th){
    Mat result = cvCreateMat(th, tw, CV_8UC1);
    float stepw = (float)(iw - 1)/(tw - 1);
    float steph = (float)(ih - 1)/(th - 1);
    result.at<uchar>(0, 0) = source.at<uchar>(0, 0);
    result.at<uchar>(0, tw - 1) = source.at<uchar>(0, iw - 1);
    result.at<uchar>(th - 1, 0) = source.at<uchar>(ih - 1, 0);
    result.at<uchar>(th - 1, tw - 1) = source.at<uchar>(ih - 1, iw - 1);
    for(int i = 0; i < th; i++){
        for(int j = 0; j < tw; j++){
            if(((i == 0) && (j == 0)) || ((i == 0) && (j == tw - 1)) || ((i == th - 1) && (j == 0)) || ((i == th - 1) && (j == tw - 1))){
                continue;
            }
            result.at<uchar>(i, j) = 0;
            float f = 0;
            float curx = j * stepw;
            float disx = curx - floor(curx);
            float cury = i * steph;
            float disy = cury - floor(cury);
            f += (float)((1 - disx) * (1 - disy) * (source.at<uchar>((int)floor(cury), (int)floor(curx))) +
            (disx) * (1 - disy) * (source.at<uchar>((int)floor(cury), (int)ceil(curx))) +
            (1 - disx) * (disy) * (source.at<uchar>((int)ceil(cury), (int)floor(curx))) +
            (disx) * (disy) * (source.at<uchar>((int)ceil(cury), (int)ceil(curx))));
            result.at<uchar>(i, j) = (uchar)f;
        }
    }
    //imshow("Original:", source);
    //imshow("Resized", result);
    //cvWaitKey(1);
    //string savename = "/Users/Jocy/Dropbox/2016 Fall/CSI 111/Program assignment/ProgramAssignment2.1/part1/text0.75.jpg";
    //imwrite(savename, result);
    return result;
}

Mat convolve(Mat source, Mat filter, int iw, int ih, int fw, int fh){
    Mat conv = cvCreateMat(iw, ih, CV_8UC1);
    
    int samplei = 0;
    int samplej = 0;
    for (int i = 0; i < conv.rows; i++){
        for (int j = 0; j < conv.cols; j++){
            conv.at<uchar>(i, j) = 0;
            float f = 0;
            for (int m = 0; m < fh; m++){
                for (int n = 0; n < fw; n++){
                    samplei = i - fh/2 + m;
                    samplej = j - fw/2 + n;
                    if (samplei >= 0 && samplei < ih && samplej >= 0 && samplej < iw){
                        f += (float)(source.at<uchar>(samplei, samplej)) * filter.at<float>(m,n);
                    }
                }
            }
            conv.at<uchar>(i, j) += (uchar)f;
        }
    }
    //cvNamedWindow("Original:", CV_WINDOW_AUTOSIZE);
    //imshow("Original:", source);
    //cvNamedWindow("Convoluted:", CV_WINDOW_AUTOSIZE);
    //imshow("Convoluted", conv);
    //imwrite("/Users/Jocy/Dropbox/2016 Fall/CSI 111/Program assignment/ProgramAssignment1/ProgramAssignment1/convoluted1Polarcities.jpg", conv);
    //cvWaitKey(1);
    //cvDestroyAllWindows();
    return conv;
}

//Mat medFilter(Mat source, int iw, int ih, int fw, int fh){

//}

Mat convolve2(Mat source, Mat filter, int iw, int ih, int fw, int fh){
    Mat conv = cvCreateMat(iw, ih, CV_32F);
    Mat prefilter = filter;
    
    int samplei = 0;
    int samplej = 0;
    for (int i = 0; i < conv.rows; i++){
        for (int j = 0; j < conv.cols; j++){
            if(i == 0 && j == 0){
                filter = Mat(3,3, CV_32F, Scalar(-1.0/3));
                filter.at<float>(1,1) = 1;
            }else if(i == 0 && j == conv.cols - 1){
                filter = Mat(3,3, CV_32F, Scalar(-1.0/3));
                filter.at<float>(1,1) = 1;
            }else if(i == conv.rows - 1 && j == conv.cols - 1){
                filter = Mat(3,3, CV_32F, Scalar(-1.0/3));
                filter.at<float>(1,1) = 1;
            }else if(i == conv.rows - 1 && j == 0){
                filter = Mat(3,3, CV_32F, Scalar(-1.0/3));
                filter.at<float>(1,1) = 1;
            }else if(i == 0){
                filter = Mat(3,3, CV_32F, Scalar(-1.0/5));
                filter.at<float>(1,1) = 1;
            }else if(i == conv.rows - 1){
                filter = Mat(3,3, CV_32F, Scalar(-1.0/5));
                filter.at<float>(1,1) = 1;
            }else if(j == 0){
                filter = Mat(3,3, CV_32F, Scalar(-1.0/5));
                filter.at<float>(1,1) = 1;
            }else if(j == conv.cols - 1){
                filter = Mat(3,3, CV_32F, Scalar(-1.0/5));
                filter.at<float>(1,1) = 1;
            }else{
                filter = prefilter;
            }
            //cout<<filter<<endl;
            conv.at<float>(i, j) = 0;
            float f = 0;
            for (int m = 0; m < fh; m++){
                for (int n = 0; n < fw; n++){
                    samplei = i - fh/2 + m;
                    samplej = j - fw/2 + n;
                    if (samplei >= 0 && samplei < ih && samplej >= 0 && samplej < iw){
                        f += (float)(source.at<uchar>(samplei, samplej)) * filter.at<float>(m,n);
                    }
                }
            }
            conv.at<float>(i, j) += (float)f;
        }
    }
    return conv;
}

Mat convolve3(Mat source, Mat filter, int iw, int ih, int fw, int fh){
    Mat conv = cvCreateMat(iw, ih, CV_32F);
    Mat prefilter = filter;
    
    int samplei = 0;
    int samplej = 0;
    for (int i = 0; i < conv.rows; i++){
        for (int j = 0; j < conv.cols; j++){
                        //cout<<filter<<endl;
            conv.at<float>(i, j) = 0;
            float f = 0;
            int count = 0;
            for (int m = 0; m < fh; m++){
                for (int n = 0; n < fw; n++){
                    samplei = i - fh/2 + m;
                    samplej = j - fw/2 + n;
                    if (samplei >= 0 && samplei < ih && samplej >= 0 && samplej < iw){
                        f += (float)(source.at<float>(samplei, samplej));
                        count++;
                    }
                }
            }
            conv.at<float>(i, j) += (float)f/count;
        }
    }
    
    return conv;
}

Mat difference(Mat source, Mat mean, int iw, int ih, int fw, int fh){
    Mat conv = cvCreateMat(iw, ih, CV_32F);
    
    int samplei = 0;
    int samplej = 0;
    for (int i = 0; i < conv.rows; i++){
        for (int j = 0; j < conv.cols; j++){
            conv.at<float>(i, j) = 0;
            float f = 0;
            for (int m = 0; m < fh; m++){
                for (int n = 0; n < fw; n++){
                    samplei = i - fh/2 + m;
                    samplej = j - fw/2 + n;
                    if (samplei >= 0 && samplei < ih && samplej >= 0 && samplej < iw){
                        f += (float)(source.at<float>(samplei, samplej) - mean.at<float>(i, j)) * (source.at<float>(samplei, samplej) - mean.at<float>(i, j));
                    }
                }
            }
            //cout<<f<<endl;
            conv.at<float>(i, j) += (float)f;
            //cout<<conv.at<float>(i, j)<<endl;
        }
    }
    return conv;
}

Mat subsample(Mat source, int iw, int ih){
    Mat result = cvCreateMat(iw/2, ih/2, CV_8UC1);
    for(int i = 0; i < result.rows; i++){
        for(int j = 0; j < result.cols; j++){
            result.at<uchar>(i, j) = source.at<uchar>((i + 1) * 2 - 1, (j + 1) * 2 - 1);
        }
    }
    return result;
}

Vector<Mat> pyramida(Mat source, string savename, cv::Mat filter, int iw, int ih, int fw, int fh){
    int i = 0;
    Vector<Mat> pyra(9);
    pyra[0] = source;
    savename = "/Users/Jocy/Dropbox/2016 Fall/CSI 111/Program assignment/ProgramAssignment2.1/part2/text";
    string presavename = savename + "pyramidA0.jpg";
    //imwrite(presavename, source);
    for(i = 0; i < 8; i++){
        Mat preimage = convolve(source , filter, source.cols, source.rows, fw, fh);
        for(int m = 0; m < preimage.rows; m++){
            preimage.at<uchar>(m, 0) = preimage.at<uchar>(m, 0)/0.7;
            preimage.at<uchar>(m, 1) = preimage.at<uchar>(m, 1)/0.95;
            preimage.at<uchar>(m, preimage.cols - 1) = preimage.at<uchar>(m, preimage.cols - 1)/0.7;
            preimage.at<uchar>(m, preimage.cols - 2) = preimage.at<uchar>(m, preimage.cols - 2)/0.95;
        }
        preimage = convolve(preimage , filter.t(), preimage.cols, preimage.rows, fh, fw);
        for(int m = 0; m < preimage.cols; m++){
            preimage.at<uchar>(0, m) = preimage.at<uchar>(0, m)/0.7;
            preimage.at<uchar>(1, m) = preimage.at<uchar>(1, m)/0.95;
            preimage.at<uchar>(preimage.rows - 1, m) = preimage.at<uchar>(preimage.rows - 1, m)/0.7;
            preimage.at<uchar>(preimage.rows - 2, m) = preimage.at<uchar>(preimage.rows - 2, m)/0.95;
        }
        preimage = subsample(preimage, preimage.rows, preimage.cols);
        
        //string presavename = savename + "pyramidA" + to_string(i+1) + ".jpg";
        //imwrite(presavename, preimage);
        source = preimage;
        pyra[i + 1] = source;
        //cout<<preimage.rows<<endl;
        //imshow("gaussian", source);
         //cvWaitKey(0);
    }
    return pyra;
    
}

Vector<Mat> secDerivative(Mat source, string savename, cv::Mat filter, int iw, int ih, int fw, int fh){
    savename = "/Users/Jocy/Dropbox/2016 Fall/CSI 111/Program assignment/ProgramAssignment4/part1/";
    Vector<Mat> gaupyramid = pyramida(source, savename, filter,source.cols, source.rows, filter.cols, filter.rows);
    Vector<Mat> secderivative(9);
    Vector<Mat> segment(9);
    Vector<Mat> cross(9);
    Vector<Mat> mean(9);
    Vector<Mat> diff(9);
    Vector<Mat> edge(9);
    float derfilter[3][3] = {{-0.125, -0.125, -0.125}, {-0.125, 1, -0.125}, {-0.125, -0.125, -0.125}};
    cv::Mat derivative = Mat(3,3, CV_32F, derfilter);
    
    for(int i = 0; i < 9; i++){
        secderivative[i] = convolve2(gaupyramid[i], derivative, gaupyramid[i].cols, gaupyramid[i].rows, 3, 3);
        //cout<<secderivative[i]<<endl;
        string presavename = savename + "step1/textsecondderivative" + to_string(i) + ".jpg";
        imwrite(presavename, secderivative[i]);
        //cout<<secderivative[i]<<endl;
        //imshow("second derivative", secderivative[i]);
        //cvWaitKey(0);
        
        segment[i] = cvCreateMat(secderivative[i].cols, secderivative[i].rows, CV_8UC1);
        for(int n = 0; n < secderivative[i].rows; n++){
            for(int m = 0; m < secderivative[i].cols; m++){
                if(secderivative[i].at<float>(n, m) > 0){
                    segment[i].at<uchar>(n, m) = 255;
                }else{
                    segment[i].at<uchar>(n, m) = 0;
                }
            }
        }
        presavename = savename + "step2/textsegment" + to_string(i) + ".jpg";
        imwrite(presavename, segment[i]);
        //imshow("segment", segment[i]);
        //cvWaitKey(0);

        cross[i] = cvCreateMat(secderivative[i].cols, secderivative[i].rows, CV_8UC1);
        for(int n = 0; n < secderivative[i].rows; n++){
            for(int m = 0; m < secderivative[i].cols; m++){
                if(n - 1 >= 0 && m - 1 >= 0){
                    if(segment[i].at<uchar>(n-1, m-1) != segment[i].at<uchar>(n, m)){
                        cross[i].at<uchar>(n, m) = 255;
                        continue;
                    }
                }
                if(n - 1 >= 0 && m + 1 < secderivative[i].cols){
                    if(segment[i].at<uchar>(n-1, m+1) != segment[i].at<uchar>(n, m)){
                        cross[i].at<uchar>(n, m) = 255;
                        continue;
                    }
                }
                if(n + 1 < secderivative[i].rows && m + 1 < secderivative[i].cols){
                    if(segment[i].at<uchar>(n+1, m+1) != segment[i].at<uchar>(n, m)){
                        cross[i].at<uchar>(n, m) = 255;
                        continue;
                    }
                }
                if(n + 1 < secderivative[i].rows && m - 1 >= 0){
                    if(segment[i].at<uchar>(n+1, m-1) != segment[i].at<uchar>(n, m)){
                        cross[i].at<uchar>(n, m) = 255;
                        continue;
                    }
                }
                if(n - 1 >= 0){
                    if(segment[i].at<uchar>(n-1, m) != segment[i].at<uchar>(n, m)){
                        cross[i].at<uchar>(n, m) = 255;
                        continue;
                    }
                }
                if(m - 1 >= 0){
                    if(segment[i].at<uchar>(n, m-1) != segment[i].at<uchar>(n, m)){
                        cross[i].at<uchar>(n, m) = 255;
                        continue;
                    }
                }
                if(n + 1 < secderivative[i].rows){
                    if(segment[i].at<uchar>(n+1, m) != segment[i].at<uchar>(n, m)){
                        cross[i].at<uchar>(n, m) = 255;
                        continue;
                    }
                }
                if(m + 1 < secderivative[i].cols){
                    if(segment[i].at<uchar>(n, m+1) != segment[i].at<uchar>(n, m)){
                        cross[i].at<uchar>(n, m) = 255;
                        continue;
                    }
                }
                cross[i].at<uchar>(n,m) = 0;
            }
        }
        presavename = savename + "step3/textzerocross" + to_string(i) + ".jpg";
        imwrite(presavename, cross[i]);
        //imshow("zerocross", cross[i]);
        //cvWaitKey(0);
        
        cv::Mat meanfilter = Mat(5,5, CV_32F, Scalar(1.0/25));
        edge[i] = Mat(secderivative[i].cols, secderivative[i].rows, CV_8UC1, Scalar(0));
        mean[i] = convolve3 (secderivative[i], meanfilter, secderivative[i].cols, secderivative[i].rows, 5, 5);
        //imshow("mean", mean[i]);
        //cvWaitKey(0);
        //cout<<mean[i]<<endl;
        diff[i] = difference(secderivative[i], mean[i], secderivative[i].cols, secderivative[i].rows, 5, 5);
        //imshow("diff", diff[i]);
        //cvWaitKey(0);
        double min, max;
        cv::minMaxLoc(diff[i], &min, &max);
        double threshold = min + (max - min) * 0.15 ;
        //cout<<diff[i]<<endl;
        for(int n = 0; n < diff[i].rows; n++){
            for(int m = 0; m < diff[i].cols; m++){
                if(cross[i].at<uchar>(n,m) == 255 && diff[i].at<float>(n,m) >= (float)threshold){
                    edge[i].at<uchar>(n,m) = 255;
                }
            }
        }
        presavename = savename + "step4/textshreshold" + to_string(i) + ".jpg";
        imwrite(presavename, edge[i]);
        //imshow("edge", edge[i]);
        //cvWaitKey(0);

        
        

    }
    

    return secderivative;
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
    
    if(source.cols != 256 || source.rows != 256){
        source = imresize(source, source.cols, source.rows, 256, 256);
    }
    
    secDerivative(source, savename, filter, source.cols, source.rows, filter.cols, filter.rows);
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
