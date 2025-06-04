# ptx-kernels
This is my understanding of PTX and how to write, compile and execute kernels to compare performance.

# Setup
Running on Intel 12th Gen i7 CPU and RTX 3050 mobile GPU (3.5GB HBM).  
Trying to comput matmul. C = A*B. Where the size of A, B, C is `8192x8192`.

We compare the accuracy using maximum absolute error of computation and baseline numpy result. It should be within `1e-3`.

# Requirements.
GPU that supports ptx version 8.
Have a python environment with following installed.
```
pycuda
numpy
```

To run the benchmark, run the following command:
```bash
python3 matmul.py
```
TODO: implement command line options to run different
kernels.

# Performance comparision.

| Kernel Names         | Time   | Speedup |
| ---------------------|--------|---------|
| Naive Kernel         | 15.64s |  1.00   |
| Mem coalescing       | 8.62s  |  1.81   |
| sh_mem blocking      | 2.44s  |  6.40   |
| sh_mem 1d tile blking| 1.24s  |  12.62  |

# References
This repo skeleton code was inspired from [here](https://github.com/unixpickle/learn-ptx)