import cupy as cp

print("QSML GPU DIAGNOSTIC")
print("CuPy version:", cp.__version__)
print("CUDA devices:", cp.cuda.runtime.getDeviceCount())

device = cp.cuda.Device()
print("GPU:", device)

a = cp.array([10, 20, 30, 40])
b = cp.array([1, 2, 3, 4])

result = a * b

print("GPU calculation:", cp.asnumpy(result))

cp.cuda.Stream.null.synchronize()

print("GPU TEST PASSED")
