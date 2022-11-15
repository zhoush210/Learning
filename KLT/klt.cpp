#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>
#include <ctype.h>

using namespace cv;
using namespace std;

void draw_lines(Mat &image, vector<Point2f> pt1, vector<Point2f> pt2)
{
    vector<Scalar> color_lut; //颜色查找表
    RNG rng(5000);
    if (color_lut.size() < pt1.size())
    {
        for (size_t t = 0; t < pt1.size(); t++)
        {
            color_lut.push_back(Scalar(rng.uniform(0, 255), rng.uniform(0, 255),
                                       rng.uniform(0, 255)));
        }
    }
    for (size_t t = 0; t < pt1.size(); t++)
    {
        line(image, pt1[t], pt2[t], color_lut[t], 2, 8, 0);
    }
    imshow("klt img2", image);
}

int main()
{
    const int MAX_COUNT = 500;
    Size subPixWinSize(10, 10), winSize(31, 31);
    TermCriteria termcrit(TermCriteria::COUNT | TermCriteria::EPS, 20, 0.03);

    Mat img1 = imread("/home/zhoush/Documents/Learning/data/img.png");
    Mat img2 = imread("/home/zhoush/Documents/Learning/data/img2.png");

    Mat gray1, gray2;
    cvtColor(img1, gray1, COLOR_BGR2GRAY);
    cvtColor(img2, gray2, COLOR_BGR2GRAY);

    vector<Point2f> points1, points2;
    goodFeaturesToTrack(gray1, points1, MAX_COUNT, 0.01, 10, Mat(), 3, 0, 0.04);
    goodFeaturesToTrack(gray2, points2, MAX_COUNT, 0.01, 10, Mat(), 3, 0, 0.04);
    // cornerSubPix(gray1, points1, subPixWinSize, Size(-1,-1), termcrit);
    // cornerSubPix(gray2, points2, subPixWinSize, Size(-1,-1), termcrit);

    vector<uchar> status;
    vector<float> err;
    calcOpticalFlowPyrLK(img1, img2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);
    imshow("img1", img1);
    draw_lines(img2, points1, points2);
    waitKey(0);
}