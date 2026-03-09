/**
 * @file cuda_utils.cuh
 * @brief CUDA and CuRAND error-handling utilities.
 *
 * Provides assertion helpers and convenience macros for checking
 * `cudaError_t` and `curandStatus_t` return codes. On failure, prints
 * the error string with file/line information and aborts.
 */

#ifndef MPPI_CUDA_UTILS_CUH
#define MPPI_CUDA_UTILS_CUH

#include <cuda_runtime.h>
#include <curand.h>
#include <iostream>
#include <cstdio>

/**
 * @brief Assert that a CUDA API call succeeded; abort with diagnostic on failure.
 *
 * @param code   Return code from a CUDA API call.
 * @param file   Source file name (`__FILE__`).
 * @param line   Source line number (`__LINE__`).
 * @param abort  If true (default), call `exit()` on error.
 */
inline void gpuAssert(cudaError_t code, const char * file, int line, bool abort = true)
{
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) {
      exit(code);
    }
  }
}

/**
 * @brief Check for asynchronous CUDA errors (calls `cudaGetLastError`).
 *
 * @param file  Source file name (`__FILE__`).
 * @param line  Source line number (`__LINE__`).
 */
inline void __cudaCheckError(const char * file, const int line)
{
  cudaError err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
    exit(-1);
  }
}

/**
 * @brief Convert a `curandStatus_t` to a human-readable string.
 * @param code  CuRAND status code.
 * @return      Null-terminated error description.
 */
inline const char * curandGetErrorString(curandStatus_t code)
{
  switch (code) {
    case CURAND_STATUS_SUCCESS: return "No errors.";
    case CURAND_STATUS_VERSION_MISMATCH: return
        "Header file and linked library version do not match.";
    case CURAND_STATUS_NOT_INITIALIZED: return "Generator not initialized.";
    case CURAND_STATUS_ALLOCATION_FAILED: return "Memory allocation failed.";
    case CURAND_STATUS_TYPE_ERROR: return "Generator is wrong type.";
    case CURAND_STATUS_OUT_OF_RANGE: return "Argument out of range.";
    case CURAND_STATUS_LENGTH_NOT_MULTIPLE: return "Length requested is not a multple of dimension.";
    case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED: return
        "GPU does not have double precision required by MRG32k3a.";
    case CURAND_STATUS_LAUNCH_FAILURE: return "Kernel launch failure.";
    case CURAND_STATUS_PREEXISTING_FAILURE: return "Preexisting failure on library entry.";
    case CURAND_STATUS_INITIALIZATION_FAILED: return "Initialization of CUDA failed.";
    case CURAND_STATUS_ARCH_MISMATCH: return
        "Architecture mismatch, GPU does not support requested feature.";
    case CURAND_STATUS_INTERNAL_ERROR: return "Internal library error.";
    default: return "Curand Error";
  }
}

/**
 * @brief Assert that a CuRAND API call succeeded; abort with diagnostic on failure.
 *
 * @param code   Return code from a CuRAND API call.
 * @param file   Source file name (`__FILE__`).
 * @param line   Source line number (`__LINE__`).
 * @param abort  If true (default), call `exit()` on error.
 */
inline void curandAssert(curandStatus_t code, const char * file, int line, bool abort = true)
{
  if (code != CURAND_STATUS_SUCCESS) {
    fprintf(stderr, "Curandassert: %s %s %d\n", curandGetErrorString(code), file, line);
    if (abort) {
      exit(code);
    }
  }
}

/** @brief Check for asynchronous CUDA errors at the call site. */
#define CudaCheckError() __cudaCheckError(__FILE__, __LINE__)

/** @brief Wrap a CUDA API call with file/line error checking. */
#define HANDLE_ERROR(ans) {gpuAssert((ans), __FILE__, __LINE__);}

/** @brief Wrap a CuRAND API call with file/line error checking. */
#define HANDLE_CURAND_ERROR(ans) {curandAssert((ans), __FILE__, __LINE__);}

#endif  // MPPI_CUDA_UTILS_CUH
