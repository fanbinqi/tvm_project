import os
import numpy as np
import onnx
import tvm
import tvm.relay as relay
 
# def prepare_graph_lib(base_path):
#   onnx_model = onnx.load('../resnet50-v2-7.onnx')
#   input_name = "data"
#   shape_dict = {input_name: (1, 3, 224, 224)}
#   mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

#   target = "llvm"
#   with tvm.transform.PassContext(opt_level=3):
#     lib = relay.build(mod, target=target, params=params)
  
#   dylib_path = os.path.join(base_path, "resnet50.so")
#   lib.export_library(dylib_path)

def prepare_graph_lib(base_path):
  img_data = np.random.rand(1, 3, 640, 640).astype("float32") / 255
  input_name = "images"
  shape_dict = {input_name:img_data.shape}
  input_shape = img_data.shape
  model_path = "../../models/yolov5n.onnx"
  onnx_model = onnx.load(model_path)
  np.random.seed(0)
  dtype = "float32"

  mod, params = relay.frontend.from_onnx(onnx_model, shape_dict, dtype = dtype)
  compiled_lib = relay.build(mod, tvm.target.Target("llvm"), params = params)
  deploy_lib_path = os.path.join(base_path, "yolov5n.so")
  compiled_lib.export_library(deploy_lib_path)
 
if __name__ == "__main__":
  curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
  prepare_graph_lib(os.path.join(curr_path, "lib"))
  printf("model import done...")
