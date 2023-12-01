```cpp
/*
A generic Swizzle functor
0bxxxxxxxxxxxxxxxYYYxxxxxxxZZZxxxx
                              ^--^ MBase is the number of least-sig bits to keep constant
                 ^-^       ^-^     BBits is the number of bits in the mask
                   ^---------^     SShift is the distance to shift the YYY mask
                                      (pos shifts YYY to the right, neg shifts YYY to the left)

e.g. Given
0bxxxxxxxxxxxxxxxxYYxxxxxxxxxZZxxx
the result is
0bxxxxxxxxxxxxxxxxYYxxxxxxxxxAAxxx where AA = ZZ xor YY
*/

template <int BBits, int MBase, int SShift = BBits>
struct Swizzle{
    // ...
}
```

```bash
row major: 
0, 1, 2, 3, 4, 5, 6, 7, 
8, 9, 10, 11, 12, 13, 14, 15, 
16, 17, 18, 19, 20, 21, 22, 23, 
24, 25, 26, 27, 28, 29, 30, 31,

swizzled layout: 
0, 1, 2, 3, 4, 5, 6, 7, 
9, 8, 11, 10, 13, 12, 15, 14, 
18, 19, 16, 17, 22, 23, 20, 21, 
27, 26, 25, 24, 31, 30, 29, 28,
```


# Reference

1. [swizzleing modes](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#swizzling-modes)
1. https://zhuanlan.zhihu.com/p/639297098
