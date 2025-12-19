import time
from infer import SentimentPredictor
from infer_onnx import ONNXSentimentPredictor

TEXT = "This movie was surprisingly good. Great acting, solid plot, and nice pacing."

def bench(predictor, n=50):
    # warmup
    for _ in range(5):
        predictor.predict(TEXT)

    t0 = time.perf_counter()
    for _ in range(n):
        predictor.predict(TEXT)
    t1 = time.perf_counter()
    return (t1 - t0) / n

def main():
    torch_pred = SentimentPredictor(model_dir="models/distilbert_imdb_small")
    onnx_pred = ONNXSentimentPredictor()

    t_torch = bench(torch_pred)
    t_onnx = bench(onnx_pred)

    print(f"Avg latency Torch: {t_torch*1000:.2f} ms")
    print(f"Avg latency ONNX : {t_onnx*1000:.2f} ms")

if __name__ == "__main__":
    main()
