# Evalution

## PC Configuration
- OS: Ubuntu 24.04
- CPU: AMD Ryzen 9 9950X 16-Core Processor
- RAM: 64GB
- GPU: NVIDIA GeForce RTX 5090 (16GB)
- Driver Version: 590.48.01
- CUDA-toolkit: 12.8

## Metrics
- Để đánh giá toàn diện hiệu quả của mô hình (đặc biệt là sau khi tối ưu hóa qua TensorRT), chúng ta tập trung vào 3 nhóm chỉ số chính:
  + Chi phí Bộ nhớ (Memory Consumption): Đo lường mức độ chiếm dụng VRAM (GPU) và RAM (System).
    * Baseline: Bộ nhớ hệ thống trước khi khởi tạo mô hình.
    * Peak: Mức đỉnh bộ nhớ ghi nhận được trong suốt quá trình load dữ liệu và thực thi mô hình.
    * Cost: Tài nguyên thực tế mà mô hình tiêu thụ (Cost=Peak−Baseline).
  + Đơn vị tính: GigaBytes (GB).
- Thời gian Suy luận (Inference Latency): Thời gian trung bình để mô hình xử lý xong một bước suy luận. Đơn vị tính: Milliseconds (ms).
- Độ chính xác (Accuracy Fidelity): Sử dụng Cosine Similarity để so sánh sai số đầu ra giữa mô hình gốc (PyTorch) và mô hình đã qua tối ưu (TensorRT).

## Measurement Method

- Hệ thống sử dụng hai bộ công cụ theo dõi (trackers) độc lập để đảm bảo tính khách quan và chính xác cho từng loại dữ liệu:

- Bộ công cụ theo dõi tài nguyên (`ResourceTracker`): Sử dụng một luồng ẩn (Background Thread) để quét liên tục bộ nhớ hệ thống với tần suất cao (mặc định 0.01s), giúp bắt được các đỉnh nhọn (spikes) về RAM/VRAM mà các phương pháp đo snapshot thông thường dễ bỏ lỡ.

```python
class ResourceTracker:
    def __init__(self, interval: float = 0.01):
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self.peak_ram = 0

    def __enter__(self):
        # clear GPU memory stats
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            self.vram_baseline = torch.cuda.memory_allocated() / (1024**3)
        
        self.ram_baseline = self.process.memory_info().rss / (1024**3)
        self.peak_ram = self.ram_baseline
        self.stop_thread = False
        self.thread = threading.Thread(target=self._monitor_ram, daemon=True)
        self.thread.start()
        return self

    def _monitor_ram(self):
        while not self.stop_thread:
            try:
                curr = self.process.memory_info().rss / (1024**3)
                if curr > self.peak_ram: self.peak_ram = curr
            except: break
            time.sleep(self.interval)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_thread = True
        if self.thread: self.thread.join()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.vram_peak = torch.cuda.max_memory_allocated() / (1024**3)
            # after measuring, clear GPU memory stats
            torch.cuda.empty_cache() 
            torch.cuda.synchronize()
        
        self.ram_cost = self.peak_ram - self.ram_baseline

    def stats(self) -> Dict[str, Tuple[float, float, float]]:
        vram_p = getattr(self, 'vram_peak', 0)
        return {
            "vram": (self.vram_baseline, vram_p, vram_p - self.vram_baseline),
            "ram": (self.ram_baseline, self.peak_ram, self.ram_cost)
        }
```

```python
class TimeTracker:
    def __init__(self, iterations: int = 30, warmup: int = 5):
        self.iterations = iterations
        self.warmup = warmup
        self.avg_time = 0

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        return self

    def run(self, func: Callable[[], None]) -> float:
        # Warm-up
        for _ in range(self.warmup):
            func()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Timed runs
        start_time = time.perf_counter()
        for _ in range(self.iterations):
            func()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        self.avg_time = ((time.perf_counter() - start_time) / self.iterations) * 1000
        return self.avg_time

    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
```

- Bộ công cụ theo dõi thời gian (`TimeTracker`): Đo lường thời gian thực thi của riêng hàm suy luận. Để loại bỏ nhiễu do quá trình nạp kernel ban đầu (Cold Start), tracker thực hiện 5 vòng lặp khởi động (Warm-up) trước khi tính toán giá trị trung bình từ 30 lần chạy chính thức. Tất cả các thao tác trên GPU đều được đồng bộ hóa thông qua `torch.cuda.synchronize()`.
```python
    with trackers.ResourceTracker(interval=0.01) as run_tracker:
        data_config = load_data_config(args.data_config)
        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()
        dataset = LeRobotSingleDataset(
            dataset_path=DATASET_PATH,
            modality_configs=modality_config,
            video_backend=args.video_backend,
            video_backend_kwargs=None,
            transforms=None,  # We'll handle transforms separately through the policy
            embodiment_tag=EMBODIMENT_TAG,
        )
        step_data = dataset[0]

        policy = Gr00tPolicy(
            model_path=MODEL_PATH,
            embodiment_tag=EMBODIMENT_TAG,
            modality_config=modality_config,
            modality_transform=modality_transform,
            denoising_steps=args.denoising_steps,
            device=device
        )

        # Setup TensorRT engines
        if args.inference_mode == 'tensorrt':
            setup_tensorrt_engines(
                policy, args.trt_engine_path, args.vit_dtype, args.llm_dtype, args.dit_dtype
            )
        predicted_action = policy.get_action(step_data)

    with trackers.TimeTracker() as time_tracker:
        inference_time = time_tracker.run(lambda: policy.get_action(step_data))
```

- Cuối cùng là chạy đánh giá độ chính xác sau khi convert và hiển thị kết quả. Để đồng nhất về accuracy, sử dụng chung một zero noise action ban đầu cho toàn bộ quá trình.

- Quy trình đánh giá được thực hiện tuần tự qua 3 giai đoạn:
  + Giai đoạn đo bộ nhớ: Thực hiện toàn bộ quy trình từ load cấu hình, khởi tạo dataset, load trọng số mô hình cho đến bước suy luận đầu tiên trong block ResourceTracker.
  + Giai đoạn đo thời gian: Sau khi tài nguyên đã ổn định, thực hiện đo latency cho riêng hàm policy.get_action thông qua TimeTracker.
  + Giai đoạn đánh giá độ lệch (Chỉ áp dụng với TensorRT):
    * Khởi tạo một bản thực thi PyTorch làm chuẩn.
    * Khởi tạo phiên bản TensorRT với các cấu hình định dạng dữ liệu (dtype) tương ứng.
    * Đảm bảo tính đồng nhất bằng cách sử dụng chung init_actions (zero noise) cho cả hai phiên bản, sau đó tính toán điểm Cosine Similarity.
```python
    if args.inference_mode == 'tensorrt':
        policy = Gr00tPolicy(
            model_path=MODEL_PATH,
            embodiment_tag=EMBODIMENT_TAG,
            modality_config=modality_config,
            modality_transform=modality_transform,
            denoising_steps=args.denoising_steps,
            device=device
        )
        # ensure PyTorch and TensorRT have the same init_actions
        if not hasattr(policy.model.action_head, "init_actions"):
            policy.model.action_head.init_actions = torch.zeros(
                1, policy.model.action_head.action_horizon, policy.model.action_head.action_dim,
                dtype=torch.float16, device=device,
            )
        # PyTorch inference
        policy.model.action_head.get_action = partial(
            action_head_pytorch_forward, policy.model.action_head
        )
        predicted_action_torch = policy.get_action(step_data)

        # Setup TensorRT engines and run inference
        setup_tensorrt_engines(
            policy, args.trt_engine_path, args.vit_dtype, args.llm_dtype, args.dit_dtype
        )
        predicted_action_tensorrt = policy.get_action(step_data)

        # Compare predictions
        cosine_score = compare_predictions(predicted_action_tensorrt, predicted_action_torch)

        print(f'\n========== Convert TensorRT Performance {args.vit_dtype}, {args.llm_dtype}, {args.dit_dtype} ==========')
    else:
        print(f'\n========== Pytorch Performance ==========')
    print('Resource Stats:')
    print('\t+ VRAM (Baseline|Peak|Cost) (GB)', run_tracker.stats()['vram'])
    print('\t+ RAM (Baseline|Peak|Cost) (GB)', run_tracker.stats()['ram'])
    print('Inference time:', time_tracker.avg_time)
    if args.inference_mode == 'tensorrt':
        print('Cosine Similarity:', cosine_score)
```

- Sử dụng script gr00t_convert_performance.py để thực hiện đo lường tự động. Có thể linh hoạt thay đổi kiểu dữ liệu (fp16, fp8, nvfp4) cho từng module (ViT, LLM, DiT) để tìm ra điểm cân bằng tối ưu (Pareto front) giữa bộ nhớ, tốc độ và độ chính xác.
```
python deployment_scripts/gr00t_convert_performance.py --vit-dtype=<fp8, or fp16> --llm-dtype=<nvfp4, fp8, or fp16> --dit-dtype=<fp8, or fp16> --inference-mode=tensorrt
```
- Ví dụ kết quả khi chạy với `vit_dtype=fp16`, `llm_dtype=fp8`, và `dit_dtype=fp8`:
```
========== Convert TensorRT Performance fp16, fp8, fp8 ==========
Resource Stats:
        + VRAM (Baseline|Peak|Cost) (GB) (0.0, 5.092715263366699, 5.092715263366699)
        + RAM (Baseline|Peak|Cost) (GB) (1.5619010925292969, 3.5546035766601562, 1.9927024841308594)
Inference time: 12.646609466689068
Cosine Similarity: 0.9501606076955795
```
## Results
<table>
  <thead>
    <tr>
      <th rowspan="2">Type</th>
      <th colspan="3">VRAM (GB)</th>
      <th colspan="3">RAM (GB)</th>
      <th rowspan="2">Inference Time (ms)</th>
      <th rowspan="2">Cosine Similarity</th>
    </tr>
    <tr>
      <th>Baseline</th><th>Peak</th><th>Cost</th>
      <th>Baseline</th><th>Peak</th><th>Cost</th>
    </tr>
  </thead>
  <tbody>
      <tr>
        <td>Pytorch</td><td>0.0</td><td>5.21</td><td>5.21</td><td>1.55</td><td>3.55</td><td>2.0</td><td>27.7</td><td>baseline</td>
      </tr>
      <tr>
        <td>ViT(fp8)+LLM(nvfp4)+DiT(fp8)</td><td>0.0</td><td>5.093</td><td>5.093</td><td>1.55</td><td>3.55</td><td>2.0</td><td>12.6</td><td>0.72</td>
      </tr>
      <tr>
        <td>ViT(fp8)+LLM(nvfp4)+DiT(fp16)</td><td>0.0</td><td>5.093</td><td>5.093</td><td>1.55</td><td>3.56</td><td>2.01</td><td>13.6</td><td>0.74</td>
      </tr>
      <tr>
        <td>ViT(fp8)+LLM(fp8)+DiT(fp8)</td><td>0.0</td><td>5.093</td><td>5.093</td><td>1.55</td><td>3.55</td><td>2.0</td><td>12.3</td><td>0.976</td>
      </tr>
      <tr>
        <td>ViT(fp8)+LLM(fp8)+DiT(fp16)</td><td>0.0</td><td>5.093</td><td>5.093</td><td>1.55</td><td>3.55</td><td>2.0</td><td>13.7</td><td><b>0.992</b></td>
      </tr>
      <tr>
        <td>ViT(fp8)+LLM(fp16)+DiT(fp8)</td><td>0.0</td><td>5.093</td><td>5.093</td><td>1.55</td><td>3.55</td><td>2.0</td><td>12.5</td><td>0.974</td>
      </tr>
      <tr>
        <td>ViT(fp8)+LLM(fp16)+DiT(fp16)</td><td>0.0</td><td>5.093</td><td>5.093</td><td>1.55</td><td>3.55</td><td>2.0</td><td>14.2</td><td><b>0.992</b></td>
      </tr>
      <tr>
        <td>ViT(fp16)+LLM(nvfp4)+DiT(fp8)</td><td>0.0</td><td>5.093</td><td>5.093</td><td>1.55</td><td>3.55</td><td>2.0</td><td><b>12.01</b></td><td>0.968</td>
      </tr>
      <tr>
        <td>ViT(fp16)+LLM(nvfp4)+DiT(fp16)</td><td>0.0</td><td>5.093</td><td>5.093</td><td>1.55</td><td>3.55</td><td>2.0</td><td>13.37</td><td>0.987</td>
      </tr>
      <tr>
        <td>ViT(fp16)+LLM(fp8)+DiT(fp8)</td><td>0.0</td><td>5.093</td><td>5.093</td><td>1.55</td><td>3.55</td><td>2.0</td><td>12.64</td><td>0.95</td>
      </tr>
      <tr>
        <td>ViT(fp16)+LLM(fp8)+DiT(fp16)</td><td>0.0</td><td>5.093</td><td>5.093</td><td>1.55</td><td>3.55</td><td>2.0</td><td>13.28</td><td>0.99</td>
      </tr>
      <tr>
        <td>ViT(fp16)+LLM(fp8)+DiT(fp16)</td><td>0.0</td><td>5.093</td><td>5.093</td><td>1.55</td><td>3.55</td><td>2.0</td><td>13.61</td><td>0.99</td>
      </tr>
      <tr>
        <td>ViT(fp16)+LLM(fp16)+DiT(fp8)</td><td>0.0</td><td>5.093</td><td>5.093</td><td>1.55</td><td>3.55</td><td>2.0</td><td>12.5</td><td>0.977</td>
      </tr>
      <tr>
        <td>ViT(fp16)+LLM(fp16)+DiT(fp16)</td><td>0.0</td><td>5.093</td><td>5.093</td><td>1.55</td><td>3.55</td><td>2.0</td><td>13.2</td><td><b>0.993</b></td>
      </tr>
  </tbody>
</table>

# Analysis and Conclusion
Dựa trên kết quả thực nghiệm từ bảng đo lường hiệu năng, chúng ta có các nhận định sau:

## Hiệu quả tối ưu hóa (Optimization Impact)
- Thời gian suy luận (Latency): Chuyển đổi sang TensorRT giúp tăng tốc độ xử lý vượt trội, giảm độ trễ hơn 2 lần (từ 27.7ms xuống còn khoảng 12-14ms).
- Độ chính xác (Accuracy Fidelity): Việc quantization gây ra sự sụt giảm độ tương đồng, nhưng ở mức kiểm soát được. Cấu hình tối ưu nhất chỉ mất khoảng 0.007 điểm Cosine Similarity so với mô hình PyTorch gốc.
- Tài nguyên bộ nhớ (Memory footprint): Mức chiếm dụng VRAM và RAM không có sự chênh lệch đáng kể giữa các cấu hình dtype và các phiên bản convert (dao động ~0.1 GB). 

## Phân tích chi tiết theo Module (Component-wise Analysis)
- DiT (Diffusion Transformer): 
  + Độ chính xác: Kiểu dữ liệu FP16 luôn duy trì độ chính xác cao hơn FP8 từ 2% đến 4% trong mọi kịch bản.
  + Hiệu năng: Việc sử dụng FP16 khiến thời gian suy luận tăng nhẹ (~1ms).
- LLM (Large Language Model):
  + Độ chính xác: Kiểu dữ liệu nvfp4 cho thấy sự sụt giảm nghiêm trọng về chất lượng (Cosine Similarity giảm mạnh xuống vùng 0.7x). Từ mức FP8 trở lên, độ chính xác đạt trạng thái bão hòa và không còn chênh lệch đáng kể.
  + Hiệu năng: Thay đổi dtype cho LLM hầu như không tác động đến tổng thời gian suy luận trên cấu hình phần cứng hiện tại.
- ViT (Vision Transformer): ViT đóng vai trò quan trọng khi kết hợp với LLM ở mức nén thấp. Đặc biệt khi LLM ở cấu hình nvfp4, việc nâng cấp ViT lên FP16 giúp cải thiện độ chính xác tới 0.24 điểm so với FP8. Khi LLM ở mức FP8/FP16, vai trò của dtype trên ViT ít rõ rệt hơn.

# Đánh giá cấu hình tối ưu (Optimal Configuration)
- Cấu hình Cân bằng nhất (Best Overall): `ViT(fp16) + LLM(fp16) + DiT(fp16)`, cấu hình này đạt độ tương đồng cao nhất (0.993) trong khi vẫn duy trì thời gian suy luận ở mức thấp (13.2ms), nhanh gấp đôi bản gốc.
- Cấu hình Tốc độ cao nhất (High Performance): `ViT(fp16) + LLM(nvfp4) + DiT(fp8)` đạt tốc độ nhanh nhất (12.01ms) nhưng phải chấp nhận độ sai lệch lớn hơn.
- Cấu hình Nén tốt nhất (Aggressive Quantization): Cần tránh sử dụng tổ hợp ViT(fp8) + LLM(nvfp4) vì gây sụt giảm chất lượng nghiêm trọng (chỉ còn 0.72), không đảm bảo độ tin cậy cho các tác vụ điều khiển robot thực tế.
