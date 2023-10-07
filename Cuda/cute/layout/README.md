# CuTe Layout

## Examples of hierarchical layout

我们用`((_2,(_2,_2)):(_4,(_2,_1)))`这个具体的例子来看如何去“思考”嵌套的layout所描述的含义。这个layout由如下C++代码创建。

```cpp
auto layout = make_layout(make_shape(2, make_shape(2, 2)),
                          make_stride(4, make_stride(2, 1)));
print_layout(layout);
```

```bash
(2,(2,2)):(4,(2,1))
     0   1   2   3 
   +---+---+---+---+
0  | 0 | 2 | 1 | 3 |
   +---+---+---+---+
1  | 4 | 6 | 5 | 7 |
   +---+---+---+---+
```

`Layout`由`Shape`和`Stride`两个分量组成：
1. `Shape`用来描述每个轴（CuTe的文档中也叫做mode）的大小
1. `Stride`用来将`Shape`描述的这个高维空间中的坐标（CuTe文档中常用"natural coordinate"这个词）转换为1维坐标。

<p align="center">
<img src="figures/example1.png" width=90%><br>
图1. 嵌套((_2,(_2,_2)):(_4,(_2,_1)))示意图
</p>

一种比较易于理解的方式是将指定Layout的过程想象为：元素在物理内存中的存储已经确定，Layout可以指定扫描这些元素的不同顺序。

对`((_2,(_2,_2)):(_4,(_2,_1)))`这个具体的例子，图1是帮助理解这个Layout的示意图。

**我们从最外层向最内层逐层看去**。外层是一个含有两个mode的layout。红色阴影部分所标识的Layout是对第二个mode进一步进行描述，我们先忽略，简单地将第二个mode看作一个整体（一个mode），等价于图1（1）这样的一个Layout。图中纵向是最内层mode，横向是外层mode。<ins>图中***方块格子中的序号是物理内存中元素的存储方式***</ins>，默认在物理内存中存储时，最内层的mode在物理内存中连续，依次向外。<ins>***红色箭头描述的是Layout指定的元素顺序***</ins>。

我们再来看内层嵌套的Layout，如图1（2）所示。这个Layout是对内层mode的进一步细粒度描述。

## Some more examples

```bash
(4,(4,2)):(4,(1,16))
       0    1    2    3    4    5    6    7 
    +----+----+----+----+----+----+----+----+
 0  |  0 |  1 |  2 |  3 | 16 | 17 | 18 | 19 |
    +----+----+----+----+----+----+----+----+
 1  |  4 |  5 |  6 |  7 | 20 | 21 | 22 | 23 |
    +----+----+----+----+----+----+----+----+
 2  |  8 |  9 | 10 | 11 | 24 | 25 | 26 | 27 |
    +----+----+----+----+----+----+----+----+
 3  | 12 | 13 | 14 | 15 | 28 | 29 | 30 | 31 |
    +----+----+----+----+----+----+----+----+
```

Layout 可以表示出重复访问：


```bash
((2,3),4):((3,1),1)
      0   1   2   3 
    +---+---+---+---+
 0  | 0 | 1 | 2 | 3 |
    +---+---+---+---+
 1  | 3 | 4 | 5 | 6 |
    +---+---+---+---+
 2  | 1 | 2 | 3 | 4 |
    +---+---+---+---+
 3  | 4 | 5 | 6 | 7 |
    +---+---+---+---+
 4  | 2 | 3 | 4 | 5 |
    +---+---+---+---+
 5  | 5 | 6 | 7 | 8 |
    +---+---+---+---+
```

# Reference

1. [CUTLASS 3 0 Next Generation Composable and Reusable GPU Linear Algebra Library - TVMCon2023](https://www.youtube.com/watch?v=QLdUML5MCfE)
1. A Generalized Micro-kernel Abstraction for GPU Linear Algebra: [[slides]](https://www.cs.utexas.edu/users/flame/BLISRetreat2023/slides/Thakkar_BLISRetreat2023.pdf): [[video]](https://www.youtube.com/watch?v=muvkCPy3UDE)
1. [Graphene: An IR for Optimized Tensor Computations on GPUs](https://dl.acm.org/doi/pdf/10.1145/3582016.3582018)
