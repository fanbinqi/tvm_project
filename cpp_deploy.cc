#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
 
class TVMResNet {
 public:
  explicit TVMResNet(const std::string& lib_path) {
    DLDevice dev{kDLCPU, 0};
    mod_factory = tvm::runtime::Module::LoadFromFile(lib_path, "so");
    gmod = mod_factory.GetFunction("default")(dev);
    set_input = gmod.GetFunction("set_input");
    get_output = gmod.GetFunction("get_output");
    run = gmod.GetFunction("run");
    // Use the C++ API
    x = tvm::runtime::NDArray::Empty({1, 3, 224, 224}, DLDataType{kDLFloat, 32, 1}, dev);
    y = tvm::runtime::NDArray::Empty({1, 1000}, DLDataType{kDLFloat, 32, 1}, dev);
  }

  void inference(cv::Mat frame) {
    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    cv::resize(frame, frame, cv::Size(224, 224));

    cv::Mat img_float;
    frame.convertTo(img_float, CV_32F);
    x.CopyFromBytes(img_float.data, 1 * 3 * 224 * 224 * sizeof(float));

    set_input("data", x);
    run();
    get_output(0, y);

    auto result = static_cast<float*>(y->data);
    for (int i = 0; i < 10; i++)
      std::cout << result[i] << std::endl;
  }
 
 private:
  // models
  tvm::runtime::Module mod_factory;
  tvm::runtime::Module gmod;
  tvm::runtime::PackedFunc set_input;
  tvm::runtime::PackedFunc get_output;
  tvm::runtime::PackedFunc run;

  // datas
  tvm::runtime::NDArray x;
  tvm::runtime::NDArray y;
};
 
int main() {
  TVMResNet res_net("../lib/resnet50.so");
  cv::Mat frame = cv::imread("../000000001000.jpg");
  res_net.inference(frame);

  return 0;
}
