
## swizzle函数的形式

swizzle函数形式如下：

```python
def shiftr(a, s):
    return a >> s if s > 0 else shiftl(a, -s)

def shiftl(a, s):
    return a << s if s > 0 else shiftr(a, -s)

# swizzle(bits, base, shift) <-- 首先定义swizzle的参数
bit_msk = (1 << bits) - 1
yyy_msk = bit_msk << (base + max(0, shift))
zzz_msk = bit_msk << (base - min(0, shift))

# 给定offset，计算这个offset被移动到哪里
return offset ^ shiftr(offset & yyy_msk, shift)
```

## 例子：swizzle<2,0,3>

swizzle是一个重排序函数，给定一个整数表示的index，输出重排序之后的index。
在内部运算的时候用输入index的二进制表示参数与swizzle函数的计算。

以`swizzle<2,0,3>`这个swizzle函数为例: bits = 2, base = 0, shift = 3
- $\text{bits}$: mask含有的bit数。
  - 在这个例子里面，bits = 2 对应mask为 `11`
- $\text{base}$: $2^{\text{base}}$在index的二进制表示中，最后2个bit保持不变。
- $\text{shift}$: $YYY \text{mask}$移动的距离。$YYY \text{mask}$是将`bits`左移`shift`位置。
  - 在这个例子里面，$YYY \text{mask}$是将mask `11`左移3位：`11000`

给定index = 8，我们来计算8会被移动到哪里？
```python
swizzle = Swizzle（2, 0, 3）

swizzle(8) = ?
```

8的二进制表示为：1000。

1. 1000 & $YYY \text{mask}$ = 1000 & 11000 = 1000
1. 1000 >> shift = 1
1. 1 ^ index = 0001 ^ 1000 = 1001

连续 0 ~ 31 在`Swizzle(2, 0, 3)`映射下，重排序结果如下：

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

1. [What does bitwise XOR (exclusive OR) mean?](https://stackoverflow.com/questions/6398427/what-does-bitwise-xor-exclusive-or-mean)
