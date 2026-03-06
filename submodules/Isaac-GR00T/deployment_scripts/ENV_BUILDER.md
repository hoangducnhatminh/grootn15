# Setup Environment:
- Giả định Driver, Cuda-toolkits, and CuDNN đã được cài.
  - OS: Ubuntu 22.04
  - Driver version: 550.163.01
  - CuDNN: 12.4

- Cài đặt TensorRT cho hệ thống, mục đích hỗ trợ chạy build engine cho bước thứ 2. Lưu ý, cần phải xem gói TensorRT hỗ 
```
# check available TensorRT version 
apt-cache policy tensorrt
# choose and install the prefer version
sudo apt-get install tensorrt=10.1.0.27-1+cuda12.4
```

- Trong nếu trong quá trình cài TensorRT với apt có xảy ra xung đột (có thể do đã cài một số thư viện liên quan đến tensorrt trước đây), nên cân nhắc sử dụng <b>aptitude</b>. Đây là một gói quản lý quy trình thông minh hơn, nó sẽ đưa ra các giải pháp (đề xuất) để giải quyết xung đột, thay vì chỉ báo lỗi và dừng lại như <b>apt</b>:
```
sudo apt-get install aptitude

sudo aptitude install tensorrt=10.1.0.27-1+cuda12.4
```

- Kiểm tra các TensorRT packages trên hệ thống (Debian/Apt) và Python Environment (Pip). Mục đích xem thử TensorRT đã được cài trên hệ thống hay chưa.
```
dpkg-query -W tensorrt
pip list | grep tensorrt
```
## Or you can install using pip
```
sudo apt-get install \
  tensorrt=10.9.0.34-1+cuda12.8 \
  libnvinfer10=10.9.0.34-1+cuda12.8 \
  libnvinfer-plugin10=10.9.0.34-1+cuda12.8 \
  libnvinfer-vc-plugin10=10.9.0.34-1+cuda12.8 \
  libnvinfer-lean10=10.9.0.34-1+cuda12.8 \
  libnvinfer-dispatch10=10.9.0.34-1+cuda12.8 \
  libnvonnxparsers10=10.9.0.34-1+cuda12.8 \
  libnvinfer-bin=10.9.0.34-1+cuda12.8 \
  libnvinfer-dev=10.9.0.34-1+cuda12.8 \
  libnvinfer-lean-dev=10.9.0.34-1+cuda12.8 \
  libnvinfer-plugin-dev=10.9.0.34-1+cuda12.8 \
  libnvinfer-vc-plugin-dev=10.9.0.34-1+cuda12.8 \
  libnvinfer-dispatch-dev=10.9.0.34-1+cuda12.8 \
  libnvonnxparsers-dev=10.9.0.34-1+cuda12.8 \
  libnvinfer-headers-dev=10.9.0.34-1+cuda12.8 \
  libnvinfer-headers-plugin-dev=10.9.0.34-1+cuda12.8 \
  libnvinfer-samples=10.9.0.34-1+cuda12.8 \
  python3-libnvinfer=10.9.0.34-1+cuda12.8 \
  python3-libnvinfer-lean=10.9.0.34-1+cuda12.8 \
  python3-libnvinfer-dispatch=10.9.0.34-1+cuda12.8 \
  python3-libnvinfer-dev=10.9.0.34-1+cuda12.8
```

or:
```
sudo apt-get install aptitude
sudo aptitude install tensorrt=10.9.0.34-1+cuda12.8
```

```
(gr00t) rtx4@vrb-rtx4-pd:~/chinhnt/Isaac-GR00T$ dpkg-query -W tensorrt
tensorrt        10.1.0.27-1+cuda12.4
(gr00t) rtx4@vrb-rtx4-pd:~/chinhnt/Isaac-GR00T$ pip list | grep tensorrt
tensorrt_cu12                10.13.0.35
tensorrt_cu12_bindings       10.13.0.35
tensorrt_cu12_libs           10.13.0.35
```


- Khởi tạo python environment, và cài đặt các gói thư viện cần thiết. Lưu ý, nên cài pytorch có phiên bản lớn hơn 2.5 và CUDA match với hệ thống, phiên bản các gói TensorRT trên python nên gần với phiên bản các gói trên hệ thống.
```
# initialize Python environment with Conda
conda create -n gr00t python=3.10
conda activate gr00t
pip install --upgrade setuptools
pip install uv

# install dependencies
uv pip install -e .[dev]

# install Model Optimization packge
uv pip install nvidia-modelopt[all]==0.33.0

# install pytorch and flash-attention
uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.1.post4/flash_attn-2.7.1.post4+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl --no-build-isolation

# install tensorrt on python
uv pip install tensorrt-cu12-libs==10.13.0.35 tensorrt-cu12==10.13.0.35 --extra-index-url https://pypi.nvidia.com

# verify packages with restricted version
uv pip install pydantic==2.10.6 numpydantic==1.6.7 numpy==1.26.4 protobuf==4.25.1 transformers==4.51.3
```

- Kiểm tra xem `nivida-modelopt` đã được cài thành công hay chưa. Lưu ý, nếu không sử dụng Tensorflow trong các bước build engine sau đó thì không cần để ý đến các warning liên quan đến Tensorflow.
```
python -c "import modelopt.torch.quantization.extensions as ext; ext.precompile()"
```

```
(gr00t) rtx4@vrb-rtx4-pd:~/chinhnt/Isaac-GR00T$ python -c "import modelopt.torch.quantization.extensions as ext; ext.precompile()"
2026-02-04 09:37:03.219735: I tensorflow/core/util/port.cc:113] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2026-02-04 09:37:03.243224: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2026-02-04 09:37:03.243259: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2026-02-04 09:37:03.244642: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2026-02-04 09:37:03.249284: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI AVX512_BF16 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2026-02-04 09:37:03.845171: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Loading extension modelopt_cuda_ext...
Loaded extension modelopt_cuda_ext in 0.0 seconds
<module 'modelopt_cuda_ext' from '/home/rtx4/.cache/torch_extensions/py310_cu124/modelopt_cuda_ext/modelopt_cuda_ext.so'>
Loading extension modelopt_cuda_ext_fp8...
Loaded extension modelopt_cuda_ext_fp8 in 0.0 seconds
<module 'modelopt_cuda_ext_fp8' from '/home/rtx4/.cache/torch_extensions/py310_cu124/modelopt_cuda_ext_fp8/modelopt_cuda_ext_fp8.so'>
Loading extension modelopt_cuda_ext_mx...
Loaded extension modelopt_cuda_ext_mx in 0.0 seconds
<module 'modelopt_cuda_ext_mx' from '/home/rtx4/.cache/torch_extensions/py310_cu124/modelopt_cuda_ext_mx/modelopt_cuda_ext_mx.so'>
```

# Export ONNX
- Chạy để export mô hình theo ONNX:
```
python deployment_scripts/export_onnx.py
```
- Hiện tại khi chạy trên môi trường vừa thiết lập chỉ có thể export theo FP16 chưa thể export với FP8.
- Bộ như chạy export vẫn chưa tối ưu.
- ...
# Deploy Engine 