for sm 90, atomic swizzle layout are defined in:
[https://github.com/NVIDIA/cutlass/blob/main/include/cute/atom/mma_traits_sm90_gmma.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/atom/mma_traits_sm90_gmma.hpp#L75-L104)

## K-major

| Swizzle Mode | Data Type | Swizzle Function | Atomic Shape        |
|:-------------|:----------|:-----------------|:--------------------|
| 128B         | half      | `Sw<3,4,3>`      | `(_8,_64):(_64,_1)` |
| 128B         | float     | `Sw<3,4,3>`      | `(_8,_32):(_32,_1)` |
| 64B          | half      | `Sw<2,4,3>`      | `(_8,_32):(_32,_1)` |
| 64B          | float     | `Sw<2,4,3>`      | `(_8,_16):(_16,_1)` |
| 32B          | half      | `Sw<1,4,3>`      | `(_8,_16):(_16,_1)` |
| 32B          | float     | `Sw<1,4,3>`      | `(_8,_8):(_8,_1)`   |

## MN-major

| Swizzle Mode | Data Type | Swizzle Function | Atomic Shape        |
|:-------------|:----------|:-----------------|:--------------------|
| 128B         | half      | `Sw<3,4,3>`      | `(_64,_8):(_1,_64)` |
| 128B         | float     | `Sw<3,4,3>`      | `(_32,_8):(_1,_32)` |
| 64B          | half      | `Sw<2,4,3>`      | `(_32,_8):(_1,_32)` |
| 64B          | float     | `Sw<2,4,3>`      | `(_16,_8):(_1,_16)` |
| 32B          | half      | `Sw<1,4,3>`      | `(_16,_8):(_1,_16)` |
| 32B          | float     | `Sw<1,4,3>`      | `(_8,_8):(_1,_8)`   |
