#pragma once

#include <cv_bridge/cv_bridge.h>
#include <nodelet/nodelet.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Header.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

namespace tl_tvm
{
class TrafficLightClassifierNodelet : public nodelet::Nodelet
{
public:
protected:
  virtual void onInit();
  virtual void callback(const sensor_msgs::Image::ConstPtr & msg);
  virtual void subscribe();
  virtual void unsubscribe();
  virtual void getLampState(const cv::Mat & input_image);
  virtual void preProcess(cv::Mat & image, float * input_tensor, bool normalize);

  ros::Publisher pub_;
  ros::Subscriber sub_;
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;

  // tvm
  // tvm::runtime::Module mod_;
  //   DLTensor * x_;
  //   DLTensor * y_;

  std::vector<float> mean_{0.242, 0.193, 0.201};
  std::vector<float> std_{1.0, 1.0, 1.0};
  int input_c_;
  int input_h_;
  int input_w_;

  int in_ndim_ = 4;
  //   int64_t in_shape_[4] = {1, input_c_, input_h_, input_w_};
  int64_t in_shape_[4] = {1, 3, 224, 224};
  int nbytes_float32_ = 4;
  int dtype_code_ = kDLFloat;
  int dtype_bits_ = 32;
  int dtype_lanes_ = 1;
  int device_cpu_ = kDLCPU;
  int device_gpu_ = kDLGPU;
  int device_cpu_id_ = 0;
  int device_gpu_id_ = 0;
  int out_ndim_ = 2;
  int64_t out_shape_[2] = {
    1,
    6,
  };

private:
  void * handle_;
};

}  // namespace tl_tvm