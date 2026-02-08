## 2026-02-08 - Thrust Functors Scope
**Learning:** Defining custom functors (structs) inside a function body for use with Thrust algorithms (`thrust::transform`) can cause cryptic `nvcc` compilation errors related to `__device_stub__` generation, even with C++17 enabled. This seems to be an issue with how `nvcc` parses local types in template arguments for device code.
**Action:** Always define functors at namespace scope or as static members of a class when using them with Thrust algorithms in `.cu` files.
