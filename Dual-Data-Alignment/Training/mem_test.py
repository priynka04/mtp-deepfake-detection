import sys, time, io, contextlib, gc
import torch
from types import SimpleNamespace
sys.path.insert(0, ".")
from networks.trainer import Trainer

def batch(n):
    r = lambda k: torch.randn(k, 3, 336, 336)
    return {"real": r(n), "real_resized": r(2*n), "fake": r(n), "fake_resized": r(2*n)}

def test(n, steps=6):
    opt = SimpleNamespace(name="memtest", checkpoints_dir="/DATA/storage/ab_students/align_gemini_project/tmp_memtest", gpu_ids=[0], lora_rank=8, lora_alpha=1.0, optim="adam", lr=1e-4, weight_decay=0.0, accumulation_steps=4)
    with contextlib.redirect_stdout(io.StringIO()):
        t = Trainer(opt)
    t.train()
    torch.cuda.reset_peak_memory_stats()
    times = []
    try:
        for i in range(steps):
            torch.cuda.synchronize(); s = time.time()
            t.set_input(batch(n)); t.optimize_parameters()
            torch.cuda.synchronize(); times.append(time.time() - s)
        print("samples/step", n, "images", 6*n, "peak GiB", round(torch.cuda.max_memory_allocated()/2**30, 1), "step s", round(sum(times[1:])/len(times[1:]), 2))
    except torch.cuda.OutOfMemoryError:
        print("samples/step", n, "images", 6*n, "OUT OF MEMORY")
    del t; gc.collect(); torch.cuda.empty_cache()

for n in (2, 1):
    test(n)
