import time
import argparse
import numpy as np
from scipy.signal import convolve2d, correlate2d
import importlib.util
import os
import sys

def try_import(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None

def synthetic_workload(batch_size, iters, in_h, in_w, k1, k2, stride1, stride2):
    rng = np.random.RandomState(0)
    x = rng.randn(batch_size, in_h, in_w).astype(float)
    kA = rng.randn(*k1).astype(float)
    kB = rng.randn(*k2).astype(float)
    # forward / backward timings
    fwd_times = []
    bwd_times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        # conv1: valid correlate -> downsample by stride1
        conv1_out = []
        for b in range(batch_size):
            corr = correlate2d(x[b], kA, mode='valid')
            conv1_out.append(corr[::stride1, ::stride1])
        conv1_out = np.stack(conv1_out)

        # conv2: use conv1_out as input
        conv2_out = []
        for b in range(batch_size):
            corr = correlate2d(conv1_out[b], kB, mode='valid')
            conv2_out.append(corr[::stride2, ::stride2])
        conv2_out = np.stack(conv2_out)

        # flatten + small dense sequence
        flat = conv2_out.reshape(batch_size, -1)
        w1 = rng.randn(flat.shape[1], 128).astype(float)
        w2 = rng.randn(128, 32).astype(float)
        w3 = rng.randn(32, 2).astype(float)
        h1 = flat.dot(w1)
        h2 = h1.dot(w2)
        out = h2.dot(w3)
        t1 = time.perf_counter()
        fwd_times.append(t1 - t0)

        # fake backward: gradients propagate through mats
        t0 = time.perf_counter()
        dout = np.ones_like(out)
        dh2 = dout.dot(w3.T)
        dh1 = dh2.dot(w2.T)
        dflat = dh1.dot(w1.T)
        # reshape and compute simple conv-backprop for conv2 and conv1 (approx via correlate)
        dconv2 = dflat.reshape(conv2_out.shape)
        # upsample and convolve with rotated kernels
        for b in range(batch_size):
            _ = correlate2d(dconv2[b], np.rot90(kB, 2), mode='full')
        # likewise for conv1
        t1 = time.perf_counter()
        bwd_times.append(t1 - t0)

    print(f"Synthetic workload: batch={batch_size} iters={iters}")
    print(f"Avg forward time: {np.mean(fwd_times):.4f}s, median: {np.median(fwd_times):.4f}s")
    print(f"Avg backward time: {np.mean(bwd_times):.4f}s, median: {np.median(bwd_times):.4f}s")
    print(f"Total time: {sum(fwd_times)+sum(bwd_times):.4f}s")

def model_workload(batch_size, iters, conv_mod, neural_mod):
    rng = np.random.RandomState(0)
    # instantiate layers similarly to main.py
    try:
        ConvLayer = getattr(conv_mod, 'ConvLayer')
        NeuralLayer = getattr(neural_mod, 'NeuralLayer')
    except Exception:
        print("Model classes not found; aborting model workload.")
        return False

    # Build layers and override kernels to random values to avoid DB access
    c1 = ConvLayer(1, 5)
    c2 = ConvLayer(2, 3)
    # set random kernels directly
    try:
        import numpy as np
        c1.kernel = rng.randn(7, 7).astype(float)  # example size
        c2.kernel = rng.randn(5, 5).astype(float)
        c1.filter_derivatives = np.zeros_like(c1.kernel)
        c2.filter_derivatives = np.zeros_like(c2.kernel)
    except Exception:
        pass

    n1 = NeuralLayer(1)
    n2 = NeuralLayer(2)

    # prepare random inputs; choose sizes so convPass likely succeeds for typical kernels
    in_h = 128
    in_w = 64
    x_batch = [rng.randn(in_h, in_w).astype(float) for _ in range(batch_size)]

    fwd_times = []
    bwd_times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        # attempt to call user convPass; fall back to synthetic slice if it raises
        try:
            out1 = c1.convPass(x_batch)
            c1.inputs = x_batch
            out2 = c2.convPass(out1)
            c2.inputs = out1
            # set outputs if required by backprop
            c1.output = out1
            c2.output = out2
            # neural forward - try to use batch_layer_output if exists, else synthetic
            try:
                flat = n1.batch_layer_output(out2)
            except Exception:
                flat = np.array(out2).reshape(batch_size, -1)
            t1 = time.perf_counter()
            fwd_times.append(t1 - t0)
        except Exception:
            # on any failure, abort model workload and return False
            print("Model forward failed - falling back to synthetic workload.")
            return False

        # try backpropagate on conv layers if exists
        t0 = time.perf_counter()
        try:
            # create fake upstream gradient matching conv2.output shape
            up_grad = np.random.randn(*np.array(c2.output).shape).astype(float)
            g1 = c2.backpropagate(up_grad, flattened=False)
            g2 = c1.backpropagate(g1, flattened=False)
            t1 = time.perf_counter()
            bwd_times.append(t1 - t0)
        except Exception:
            print("Model backprop failed - falling back to synthetic workload.")
            return False

    print(f"Model workload: batch={batch_size} iters={iters}")
    print(f"Avg forward time: {np.mean(fwd_times):.4f}s")
    print(f"Avg backward time: {np.mean(bwd_times):.4f}s")
    print(f"Total time: {sum(fwd_times)+sum(bwd_times):.4f}s")
    return True

def main():
    parser = argparse.ArgumentParser(description="Benchmark model or synthetic workload")
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--iters', type=int, default=20)
    parser.add_argument('--use-model', action='store_true', help='Attempt to run real model code (may require DB etc.)')
    args = parser.parse_args()

    project_dir = os.path.dirname(__file__)
    conv_path = os.path.join(project_dir, "ConvolutionLayer.py")
    neural_path = os.path.join(project_dir, "NeuralLayer.py")

    conv_mod = try_import('convmod', conv_path)
    neural_mod = try_import('neuralmod', neural_path)

    if args.use_model and conv_mod is not None and neural_mod is not None:
        ok = model_workload(args.batch, args.iters, conv_mod, neural_mod)
        if ok:
            return
        else:
            print("Falling back to synthetic workload.")
    else:
        if args.use_model:
            print("Could not import model modules; falling back to synthetic workload.")

    # synthetic params roughly matching your architecture
    in_h = 128
    in_w = 64
    k1 = (7, 7)
    k2 = (5, 5)
    stride1 = 5
    stride2 = 3
    synthetic_workload(args.batch, args.iters, in_h, in_w, k1, k2, stride1, stride2)

if __name__ == '__main__':
    main()