#include <cstdio>
#include <fstream>
#include <vector>
#include <algorithm>
#include <iostream>
#include <typeinfo>
#include <algorithm>

#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

typedef struct BoxInfo {
  float x1;
  float y1;
  float x2;
  float y2;
  float score;
  int label;
} BoxInfo;

class Yolov5 {
 public:
  explicit Yolov5(const std::string &lib_path) {
    LOG(INFO) << "Running Graph Executor...";
    DLDevice dev{kDLCPU, 0};
    mod_factory = tvm::runtime::Module::LoadFromFile(lib_path, "so");
    gmod = mod_factory.GetFunction("default")(dev);
    set_input = gmod.GetFunction("set_input");
    get_output = gmod.GetFunction("get_output");
    run = gmod.GetFunction("run");

    int in_dim = 4;
    int out_ndim = 3;
    int64_t in_shape[4] = {1, 3, 640, 640};
    int64_t out_shape[3] = {1, 25200, 85};
    int dtype_code = kDLFloat;
    int dtype_bits = 32;
    int dtype_lanes = 1;
    int device_type = kDLCPU;
    int device_id = 0;
    TVMArrayAlloc(out_shape, out_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &y);
    TVMArrayAlloc(in_shape, in_dim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &x);
  }

  void inference(cv::Mat &frame) {
    cv::Mat in_put;
    cv::resize(frame, in_put, cv::Size(640, 640));
    float img_data[640 * 640 * 3];
    Mat_to_CHW(img_data, in_put);
    memcpy(x->data, &img_data, 3 * 640 * 640 * sizeof(float));

    set_input("images", x);
    run();
    get_output(0, y);

    static float result[25200][85] = {0};
    TVMArrayCopyToBytes(y, result, 25200 * 85 * sizeof(float));
    int num_proposal = sizeof(result) / sizeof(result[0]);      // 25200
    int box_classes = sizeof(result[0]) / sizeof(result[0][0]); // 85
    std::cout << "num_proposal: " << num_proposal << std::endl;
    std::cout << "box_classes: " << box_classes << std::endl;

    std::vector<BoxInfo> generate_boxes;
    float *pdata = result[0];
    float padw = 0, padh = 0;
    float ratioh = 1, ratiow = 1;
    for (int i = 0; i < num_proposal; ++i) {
      int index = i * box_classes;
      float obj_conf = pdata[index + 4];
      if (obj_conf > 0.2) {
        std::cout << "obj_conf" << obj_conf << std::endl;
        int class_idx = 0;
        float max_class_socre = 0;
        for (int k = 0; k < 80; ++k) {
          if (pdata[k + index + 5] > max_class_socre) {
            max_class_socre = pdata[k + index + 5];
            class_idx = k;
          }
        }
        if (max_class_socre > 0.6) {
          float cx = pdata[index];
          float cy = pdata[index + 1];
          float w = pdata[index + 2];
          float h = pdata[index + 3];
          float xmin = ((cx - padw - 0.5 * w) * ratiow); // *ratiow，变回原图尺寸
          float ymin = ((cy - padh - 0.5 * h) * ratioh);
          float xmax = (cx - padw + 0.5 * w) * ratiow;
          float ymax = (cy - padh + 0.5 * h) * ratioh;
          generate_boxes.push_back(BoxInfo{xmin, ymin, xmax, ymax, max_class_socre, class_idx});
        }
      }
    }
    nms(generate_boxes);
    std::cout << generate_boxes.size() << std::endl;
    for (size_t i = 0; i < generate_boxes.size(); i++) {
      float xmin = generate_boxes[i].x1;
      float xmax = generate_boxes[i].x2;
      float ymin = generate_boxes[i].y1;
      float ymax = generate_boxes[i].y2;
      float score = generate_boxes[i].score;
      int classes = generate_boxes[i].label;
      cv::rectangle(in_put, cv::Point(xmin, ymin), cv::Point(int(generate_boxes[i].x2), int(generate_boxes[i].y2)), cv::Scalar(0, 0, 255), 2);
      std::string label = cv::format("%.2f", generate_boxes[i].score);
      cv::putText(in_put, label, cv::Point(xmin, ymin - 5), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 0), 1);
      std::cout << "xmin:" << xmin << std::endl;
      std::cout << "xmax:" << xmax << std::endl;
      std::cout << "ymin:" << ymin << std::endl;
      std::cout << "ymax:" << ymax << std::endl;
      std::cout << "score:" << score << std::endl;
      std::cout << "classes:" << classes << std::endl;
    }
    std::cout << "----------" << std::endl;
    cv::imwrite("./result.jpg", in_put);
  }

 private:
  void nms(std::vector<BoxInfo> &input_boxes) {
    float nmsThreshold = 0.45;
    std::sort(input_boxes.begin(), input_boxes.end(),
              [](BoxInfo a, BoxInfo b)
              { return a.score > b.score; });
    std::vector<float> vArea(input_boxes.size());
    for (int i = 0; i < input_boxes.size(); ++i) {
      vArea[i] = (input_boxes[i].x2 - input_boxes[i].x1 + 1) *
                 (input_boxes[i].y2 - input_boxes[i].y1 + 1);
    }

    std::vector<bool> isSuppressed(input_boxes.size(), false);
    for (int i = 0; i < input_boxes.size(); ++i) {
      if (isSuppressed[i]) {
        continue;
      }
      for (int j = i + 1; j < input_boxes.size(); ++j) {
        if (isSuppressed[j]) {
          continue;
        }
        float xx1 = std::max(input_boxes[i].x1, input_boxes[j].x1);
        float yy1 = std::max(input_boxes[i].y1, input_boxes[j].y1);
        float xx2 = std::min(input_boxes[i].x2, input_boxes[j].x2);
        float yy2 = std::min(input_boxes[i].y2, input_boxes[j].y2);
        float w = std::max(0.0f, xx2 - xx1 + 1);
        float h = std::max(0.0f, yy2 - yy1 + 1);
        float inter = w * h;

        if (input_boxes[i].label == input_boxes[j].label) {
          float ovr = inter / (vArea[i] + vArea[j] - inter);
          if (ovr >= nmsThreshold) {
            isSuppressed[j] = true;
          }
        }
      }
    }
    int idx_t = 0;
    input_boxes.erase(std::remove_if(input_boxes.begin(), input_boxes.end(), [&idx_t, &isSuppressed](const BoxInfo &f)
                                     { return isSuppressed[idx_t++]; }),
                      input_boxes.end());
  }

  void Mat_to_CHW(float *img_data, cv::Mat &frame) {
    assert(img_data && !frame.empty());
    unsigned int volChl = 640 * 640;

    for (int c = 0; c < 3; ++c) {
      for (unsigned j = 0; j < volChl; ++j)
        img_data[c * volChl + j] = static_cast<float>(float(frame.data[j * 3 + c]) / 255.0);
    }
  }

 private:
  tvm::runtime::Module mod_factory;
  tvm::runtime::Module gmod;
  tvm::runtime::PackedFunc set_input;
  tvm::runtime::PackedFunc get_output;
  tvm::runtime::PackedFunc run;

  DLTensor *x;
  DLTensor *y;
};

int main() {
  Yolov5 net("../lib/yolov5n.so");
  cv::Mat frame = cv::imread("../000000001000.jpg");
  net.inference(frame);

  return 0;
}
