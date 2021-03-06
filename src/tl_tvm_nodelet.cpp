#include "tl_tvm/tl_tvm_nodelet.h"
#include <stdio.h>
#include <chrono>
#include <fstream>

namespace tl_tvm
{
void TrafficLightClassifierNodelet::onInit()
{
  nh_ = getNodeHandle();
  pnh_ = getPrivateNodeHandle();
  pub_ = pnh_.advertise<sensor_msgs::Image>("output", 1);

  TrafficLightClassifierNodelet::subscribe();

  std::string label_file_path;
  std::string model_file_path;
  std::string model_json_path;
  std::string model_params_path;

  pnh_.param<std::string>("label_file_path", label_file_path, "labels.txt");
  pnh_.param<std::string>("model_file_path", model_file_path, "data/mobilenetv2.so");
  pnh_.param<std::string>("model_json_path", model_json_path, "data/model_graph.json");
  pnh_.param<std::string>("model_params_path", model_params_path, "data/model_graph.params");

  pnh_.param<int>("input_c", input_c_, 3);
  pnh_.param<int>("input_h", input_h_, 224);
  pnh_.param<int>("input_w", input_w_, 224);
  in_shape_[0] = 1;
  in_shape_[1] = input_c_;
  in_shape_[2] = input_h_;
  in_shape_[3] = input_w_;

  readLabelfile(label_file_path, labels_);

  std::cout << model_file_path << std::endl;
  tvm::runtime::Module mobilenet_lib_ = tvm::runtime::Module::LoadFromFile(model_file_path);

  std::ifstream json_in(model_json_path, std::ios::in);
  std::string json_data(
    (std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
  json_in.close();

  std::ifstream params_in(model_params_path, std::ios::binary);
  std::string params_data(
    (std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
  params_in.close();

  TVMByteArray params_arr;
  params_arr.data = params_data.c_str();
  params_arr.size = params_data.length();

  tvm::runtime::Module mod = (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))(
    json_data, mobilenet_lib_, device_gpu_, device_gpu_id_);
  this->handle_ = new tvm::runtime::Module(mod);

  tvm::runtime::PackedFunc load_params = mod.GetFunction("load_params");
  load_params(params_arr);
}

void TrafficLightClassifierNodelet::subscribe()
{
  sub_ = pnh_.subscribe("input", 1, &TrafficLightClassifierNodelet::callback, this);
}

void TrafficLightClassifierNodelet::unsubscribe() { sub_.shutdown(); }

bool TrafficLightClassifierNodelet::readLabelfile(
  std::string filepath, std::vector<std::string> & labels)
{
  std::ifstream labelsFile(filepath);
  if (!labelsFile.is_open()) {
    ROS_ERROR("Could not open label file. [%s]", filepath.c_str());
    return false;
  }
  std::string label;
  while (getline(labelsFile, label)) {
    labels.push_back(label);
  }
  return true;
}

void TrafficLightClassifierNodelet::preProcess(
  cv::Mat & image, float * input_tensor, bool normalize)
{
  // cv::cvtColor(image, image, cv::COLOR_BGR2RGB, 3);
  cv::resize(image, image, cv::Size(input_w_, input_h_));

  const size_t strides_cv[3] = {static_cast<size_t>(input_w_ * input_c_),
                                static_cast<size_t>(input_c_), 1};
  const size_t strides[3] = {static_cast<size_t>(input_h_ * input_w_),
                             static_cast<size_t>(input_w_), 1};

  for (int i = 0; i < input_h_; i++) {
    for (int j = 0; j < input_w_; j++) {
      for (int k = 0; k < input_c_; k++) {
        const size_t offset_cv = i * strides_cv[0] + j * strides_cv[1] + k * strides_cv[2];
        const size_t offset = k * strides[0] + i * strides[1] + j * strides[2];
        if (normalize) {
          input_tensor[offset] = (((float)image.data[offset_cv] / 255) - mean_[k]) / std_[k];
        } else {
          input_tensor[offset] = (float)image.data[offset_cv];
        }
      }
    }
  }
}

void TrafficLightClassifierNodelet::getLampState(const cv::Mat & input_image)
{
  DLTensor * x;
  DLTensor * y;

  float * input_data = (float *)malloc(input_w_ * input_h_ * input_c_ * sizeof(float));
  cv::Mat image = input_image.clone();

  TVMArrayAlloc(
    in_shape_, in_ndim_, dtype_code_, dtype_bits_, dtype_lanes_, device_cpu_, device_cpu_id_, &x);
  x->data = input_data;
  preProcess(image, input_data, true);

  tvm::runtime::Module * mod = (tvm::runtime::Module *)handle_;
  tvm::runtime::PackedFunc set_input = mod->GetFunction("set_input");

  set_input("input", x);

  TVMArrayAlloc(
    out_shape_, out_ndim_, dtype_code_, dtype_bits_, dtype_lanes_, device_cpu_, device_cpu_id_, &y);

  tvm::runtime::PackedFunc run = mod->GetFunction("run");
  run();

  tvm::runtime::PackedFunc get_output = mod->GetFunction("get_output");

  get_output(0, y);
  printf("\noutput\n----------\n");

  float max_label = -100;
  int max_label_id = 0;
  for (int i = 0; i < 6; ++i) {
    if (static_cast<float *>(y->data)[i] > max_label) {
      max_label = static_cast<float *>(y->data)[i];
      max_label_id = i;
    }
    std::cout << labels_[i] << ": " << static_cast<float *>(y->data)[i] << std::endl;
  }
  printf("----------\n");
  std::cout << "result: " << labels_[max_label_id] << std::endl;

  TVMArrayFree(x);
  TVMArrayFree(y);
}

void TrafficLightClassifierNodelet::callback(const sensor_msgs::Image::ConstPtr & input_image_msg)
{
  std::chrono::system_clock::time_point start, end;
  start = std::chrono::system_clock::now();

  cv_bridge::CvImagePtr cv_ptr;

  try {
    cv_ptr = cv_bridge::toCvCopy(input_image_msg, sensor_msgs::image_encodings::RGB8);
    cv::Mat hoge = cv_ptr->image;

  } catch (cv_bridge::Exception & e) {
    NODELET_ERROR("Could not convert from '%s' to 'rgb8'.", input_image_msg->encoding.c_str());
  }

  getLampState(cv_ptr->image);

  end = std::chrono::system_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

  printf("time %lf[ms]\n", elapsed);
}

}  // namespace tl_tvm

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(tl_tvm::TrafficLightClassifierNodelet, nodelet::Nodelet);