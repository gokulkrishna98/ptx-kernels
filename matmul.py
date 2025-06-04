from typing import Any, Callable
import numpy as np
from context import compile_function, gpu_to_numpy, measure_time, numpy_to_gpu, sync

"""
NOTE: We perform matrix multiplication on 8192*8192  matrix
which is equivalent to 2^26 elements, i.e appx 10^8 elements per matrix
"""

"""
This matmul creates block of 32x32 and performs divides the grid into
256(= 8192/32)x 256 blocks.
"""
def evaluate_matmul_fn(fn: Callable):
    def call_fn(A: np.ndarray, B: np.ndarray, A_buf: Any, B_buf: Any, out_buf: Any):
        block_size = 32
        fn(
            A_buf,
            B_buf,
            out_buf,
            np.int32(A.shape[0] // block_size),
            grid=(
                A.shape[0] // block_size,
                A.shape[1] // block_size,
                1,
            ),
            block=(block_size, block_size, 1),
        )

    generic_eval_matmul(call_fn)


def generic_eval_matmul(fn: Callable, block_mult: int = 1):
    size = 8192
    A = np.random.normal(size=[size, size]).astype(np.float32)
    B = np.random.normal(size=[size, size]).astype(np.float32)

    A_buf = numpy_to_gpu(A)
    B_buf = numpy_to_gpu(B)
    out_buf = numpy_to_gpu(A * 0)
    with measure_time() as timer:
        fn(
            A,
            B,
            A_buf,
            B_buf,
            out_buf,
        )
    sync()
    results = gpu_to_numpy(out_buf, A.shape, A.dtype)
    expected = A @ B

    # print(f"expected:\n {expected}\n")
    # print(f"A:\n {A}\n")
    # np.set_printoptions(threshold=np.inf)
    # print(f"B:\n {B}\n")
    # print(f"results:\n {results}\n")

    print(f"maximum absolute error of matmul is {np.abs(results - expected).max()}")
    print(f"time elapsed: {timer()}")

def matmul_simple():
    fn = compile_function("sgemm_naive.ptx", "sgemm_naive")
    evaluate_matmul_fn(fn)

def matmul_mem_coalesce():
    fn = compile_function("sgemm_mem_coalesce.ptx", "sgemm_mem_coalesce")
    evaluate_matmul_fn(fn)

def matmul_shmem_blocking():
    fn = compile_function("sgemm_shmem_blocking.ptx", "sgemm_shmem_blocking")
    evaluate_matmul_fn(fn)

if __name__ == "__main__":
    # matmul_simple()
    # matmul_mem_coalesce()
    matmul_shmem_blocking()

