## 2026-02-07 - [CUDA Kernel Optimization with SFINAE]
**Learning:** Using SFINAE to detect compile-time constants (like `STATE_DIM`) allows specializing CUDA kernels without explicit template specialization for every case. `if constexpr` combined with `std::void_t` detection enables cleaner code that supports both static and dynamic loop bounds in a single kernel template.
**Action:** Use this pattern to auto-detect optimization opportunities in templated kernels instead of forcing users to pass template parameters manually.
