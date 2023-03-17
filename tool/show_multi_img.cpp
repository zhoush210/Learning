#include <iostream>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

vector<Mat> imageVector;

void multipleImage(vector<Mat> imgVector, Mat& dst, int imgCols) 
{
    const int MAX_PIXEL=900;
    int imgNum = imgVector.size();
    //选择图片最大的一边 将最大的边按比例变为300像素
    Size imgOriSize = imgVector[0].size();
    int imgMaxPixel = max(imgOriSize.height, imgOriSize.width);
    //获取最大像素变为MAX_PIXEL的比例因子
    double prop = imgMaxPixel < MAX_PIXEL ?  (double)imgMaxPixel/MAX_PIXEL : MAX_PIXEL/(double)imgMaxPixel;
    Size imgStdSize(imgOriSize.width * prop, imgOriSize.height * prop); //窗口显示的标准图像的Size

    Mat imgStd; //标准图片
    Point2i location(0, 0); //坐标点(从0,0开始)
    //构建窗口大小 通道与imageVector[0]的通道一样
    Mat imgWindow(imgStdSize.height*((imgNum-1)/imgCols + 1), imgStdSize.width*imgCols, imgVector[0].type());
    for (int i=0; i<imgNum; i++)
    {
        location.x = (i%imgCols)*imgStdSize.width;
        location.y = (i/imgCols)*imgStdSize.height;
        resize(imgVector[i], imgStd, imgStdSize, prop, prop, INTER_LINEAR); //设置为标准大小
        imgStd.copyTo( imgWindow( Rect(location, imgStdSize) ) );
    }
    dst = imgWindow;
}

int main()
{
    Mat dst;
    Mat image1 = imread("/home/zhoush/Downloads/1.jpg");
    Mat image2 = imread("/home/zhoush/Downloads/2.jpg");
    Mat image3 = imread("/home/zhoush/Downloads/3.jpg");
    Mat image4 = imread("/home/zhoush/Downloads/4.jpg");

    imageVector.push_back(image1);
    imageVector.push_back(image2);
    imageVector.push_back(image3);
    imageVector.push_back(image4);

    multipleImage(imageVector, dst, 4);

    namedWindow("multipleWindow");
    imshow("multipleWindow", dst);
    imwrite("1.png", dst);
    waitKey(0);
    destroyAllWindows();
}

