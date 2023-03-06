#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <regex>
using namespace std;

// 画图参考：https://blog.csdn.net/qq_33548747/article/details/88135796
int main()
{
    ifstream f("/home/zhoush/Documents/test/draw/01xieshi.txt");
    string line;
    regex pose_regex("(.*),(.*),(.*)"); // 正则匹配
    std::vector<cv::Point> points;
    while (getline(f, line))
    {
        smatch result;
        bool find_pose = regex_match(line, result, pose_regex);
        if (find_pose)
        {
            float x, y;
            string x_str, y_str;
            x_str = result[1].str();
            y_str = result[2].str();
            x = atof(x_str.c_str());
            y = atof(y_str.c_str());
            points.push_back(cv::Point(10 * x + 200, 10 * y + 200));
        }
    }

    // 创建用于绘制的背景图像
    cv::Mat image = cv::Mat::zeros(480, 640, CV_8UC3);
    image.setTo(cv::Scalar(255, 255, 255));

    // 将点绘制到空白图上
    for (int i = 0; i < points.size(); i++)
    {
        cv::circle(image, points[i], 5, cv::Scalar(0, 0, 255), 2, 8, 0);
    }
    // 绘制折线
    cv::polylines(image, points, false, cv::Scalar(0, 255, 0), 1, 8, 0);

    cv::imshow("image", image);
    cv::imwrite("img.png", image);
    cv::waitKey(0);
}