"""저장 모델의 CPU 추론을 같은 프로토콜로 측정. 학습 메모리는 별도 범위로 기록한다."""

import platform
import statistics
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import psutil
import torch
from step06_predict import load_run


def cpu_rss_mib():
    return psutil.Process().memory_info().rss / 2**20


def measurement_environment(device="cpu"):
    cpu_name = platform.processor() or platform.machine()
    if platform.system() == "Linux":
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                cpu_name = line.split(":", 1)[1].strip()
                break
    return {"os": platform.platform(), "python": platform.python_version(),
            "hostname": platform.node(), "machine": platform.machine(), "cpu": cpu_name,
            "logical_cpus": psutil.cpu_count(), "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(), "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if str(device).startswith("cuda") else None,
            "versions": {name: version(name) for name in
                         ("numpy", "scikit-learn", "torch", "transformers", "psutil")}}


def predict_scored(text, bundle):
    """평가 어댑터: 정답을 모델에 전달하지 않는다. 확률 없는 sklearn 분류기도 지원한다.

    서비스와 같은 strip·토큰 길이 제한을 적용한다. 서비스의 입력 검사·절단 여부 검사는 제외한다.
    새 모델도 {prediction: int, probabilities: list | None} 계약으로 평가 코드를 재사용할 수 있다.
    """
    text = text.strip()
    model = bundle["model"]
    if bundle["kind"] == "baseline":
        if not hasattr(model, "predict_proba"):
            return {"prediction": int(model.predict([text])[0]), "probabilities": None}
        scores = model.predict_proba([text])[0]
    else:
        features = bundle["tokenizer"](
            text, truncation=True, max_length=bundle["max_length"], return_tensors="pt")
        with torch.inference_mode():
            scores = model(**features).logits.softmax(-1)[0].cpu().numpy()
    return {"prediction": int(np.argmax(scores)), "probabilities": scores.tolist()}


def benchmark_predict(predict, texts, *, warmup=1, repeats=1):
    """CPU 동기식 콜백; 샘플별 시간을 직접 관찰하고 각 pass 전체 시간도 측정한다."""
    if not texts or warmup < 0 or repeats < 1:
        raise ValueError("입력은 비어 있지 않아야 하며 warmup>=0, repeats>=1이어야 합니다.")
    for index in range(warmup):
        predict(texts[index % len(texts)])
    durations, pass_seconds, first_results = [], [], None
    for _ in range(repeats):
        results = []
        started = time.perf_counter()
        for text in texts:
            sample_started = time.perf_counter()
            results.append(predict(text))
            durations.append(time.perf_counter() - sample_started)
        pass_seconds.append(time.perf_counter() - started)
        if first_results is None:
            first_results = results
        elif results != first_results:
            raise ValueError("반복 추론 결과가 달라졌습니다. 평가 모드와 결정성을 확인하세요.")
    seconds = statistics.mean(pass_seconds)
    return first_results, {"inference_device": "cpu", "inference_batch_size": 1,
                           "warmup_samples": warmup, "measurement_repeats": repeats,
                           "sample_count": len(texts), "inference_total_seconds": seconds,
                           "inference_pass_seconds": pass_seconds,
                           "inference_mean_ms_per_sample": statistics.mean(durations) * 1000,
                           "inference_median_ms_per_sample": statistics.median(durations) * 1000,
                           "throughput_samples_per_second": len(texts) / seconds,
                           "inference_protocol": "cpu-single-text-v1",
                           "inference_scope": "strip + vectorization/tokenization + forward + probabilities; "
                           "excludes model loading, warmup, metrics, artifact I/O, service input checks"}


def model_resources(bundle):
    model = bundle["model"]
    if bundle["kind"] == "baseline":
        classifier = model[-1]
        count = int(classifier.coef_.size + classifier.intercept_.size)
        paths = [Path(bundle["run_dir"]) / "baseline.joblib"]
        definition = "LR coefficients + intercepts; TF-IDF vocabulary/IDF excluded"
    else:
        count = sum(parameter.numel() for parameter in model.parameters())
        paths = [path for path in (Path(bundle["run_dir"]) / "checkpoint").rglob("*")
                 if path.is_file()]
        definition = "all backbone and classification head parameters"
    size = sum(path.stat().st_size for path in paths)
    return {"parameter_count": count, "parameter_count_definition": definition,
            "model_size_bytes": size, "model_size_mib": size / 2**20,
            "model_size_scope": "baseline.joblib" if bundle["kind"] == "baseline"
            else "checkpoint/ weights + config + tokenizer; excludes copied baseline"}


def measure_loaded_bundle(bundle, rows, *, warmup=1, repeats=1):
    results, efficiency = benchmark_predict(lambda text: predict_scored(text, bundle),
                                            [row["text"] for row in rows],
                                            warmup=warmup, repeats=repeats)
    efficiency.update({"cpu_rss_after_inference_mib": cpu_rss_mib(),
                       "cpu_memory_scope": "process RSS snapshots, includes other live models/caches; "
                       "not isolated model memory or a sampled peak",
                       "cpu_peak_memory_mib": None,
                       "cpu_peak_memory_reason": "RSS snapshots only; peak not sampled",
                       "environment": measurement_environment(), **model_resources(bundle)})
    probabilities = [result["probabilities"] for result in results]
    if any(value is None for value in probabilities):
        probabilities = None
    return [result["prediction"] for result in results], probabilities, efficiency


def measure_saved_run(run, rows, *, warmup=1, repeats=1):
    before = cpu_rss_mib()
    started = time.perf_counter()
    bundle = load_run(run)  # 기존 로더의 라벨 검사·local_files_only·CPU eval 모드를 재사용
    load_seconds = time.perf_counter() - started
    loaded = cpu_rss_mib()
    predictions, probabilities, efficiency = measure_loaded_bundle(bundle, rows, warmup=warmup,
                                                                  repeats=repeats)
    efficiency.update({"model_load_seconds": load_seconds,
                       "cpu_rss_before_load_mib": before, "cpu_rss_after_load_mib": loaded,
                       })
    return predictions, probabilities, efficiency
