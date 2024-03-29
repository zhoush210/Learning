#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
 
using namespace std;
using namespace Eigen;
 
int main(int argc, char **argv) {
  double ar = 1.0, br = 2.0, cr = 1.0;         // real 真实参数值
  double ae = 2.0, be = -1.0, ce = 5.0;        // estimate 估计参数值,赋初值
  int N = 100;                                 // 数据点个数
  double w_sigma = 1.0;                        // 噪声Sigma值
  double inv_sigma = 1.0 / w_sigma;
  cv::RNG rng;                                 // OpenCV随机数产生器
  vector<double> x_data, y_data;      // 数据
  // 生成随机数据
  for (int i = 0; i < N; i++) {
    double x = i / 100.0;
    x_data.push_back(x);
    y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));
  }
  
  // 开始Gauss-Newton迭代
  int iterations = 100;    // 迭代次数
  double cost = 0, lastCost = 0;  // 本次迭代的cost和上一次迭代的cost
  
  for (int iter = 0; iter < iterations; iter++) {
    Matrix3d H = Matrix3d::Zero();             // Hessian = J^T W^{-1} J in Gauss-Newton
    Vector3d b = Vector3d::Zero();             // bias
    cost = 0;
    for (int i = 0; i < N; i++) {
      double xi = x_data[i], yi = y_data[i];  // 第i个数据点
      double error = yi - exp(ae * xi * xi + be * xi + ce);
      Vector3d J; // 雅可比矩阵
      J[0] = xi * xi * exp(ae * xi * xi + be * xi + ce);  // de/dae
      J[1] = xi * exp(ae * xi * xi + be * xi + ce);  // de/dbe
      J[2] = exp(ae * xi * xi + be * xi + ce);  // de/dce
      H += inv_sigma * inv_sigma * J * J.transpose();
      b += inv_sigma * inv_sigma * error * J;
      cost += error * error;       // 累加数据的误差二范数
    }
 
    // 求解线性方程 Hx=b
    Vector3d dx = H.ldlt().solve(b);
    // 无效解
    if (isnan(dx[0])) {
      cout << "result is nan!" << endl;
      break;
    }
    // 迭代值变大,则终止迭代
    if (iter > 0 && cost >= lastCost) {
      cout << "cost: " << cost << ">= last cost: " << lastCost << ", break." << endl;
      break;
    }
    // 更新解
    ae += dx[0];
    be += dx[1];
    ce += dx[2];
    lastCost = cost;
    // 记录迭代过程
    cout << "total cost: " << cost << ", \t\tupdate: " << dx.transpose() <<
         "\t\testimated params: " << ae << "," << be << "," << ce << endl;
  }
  cout << "estimated abc = " << ae << ", " << be << ", " << ce << endl;
  return 0;
}