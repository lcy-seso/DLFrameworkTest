## 一个generic swizzle函数

对一个二维空间的行和列进行swizzle $\text{swizzle}(B,M,S)$：

- $2^M$ 个元素为一组
- swizzle的二维空间有 $2^B$ 行
- swizzle的二维空间中 $2^S$ 个元素为一列

每个线程用向量化指令访问128b数据，$128 / 32 = 4 \ \text{bank}$，每个线程访问4个bank，8个线程访问一条shared memory cache line。

1. 当数据类型是半精度时，$M=3$，因为$2^3=8 \times 16 = 128 \ \text{b}$，128-bit 访存指令读取8个元素，这些元素为一组。
1. $S = 3$，1024 / 128 = 8，8个线程访问一整条shared memory cache line
1. 假如原始输入数据有形状，在内存中连续的维度是64，并且数据类型为半精度，$64 \times 16 / 1024 = 1$，一个连续维度就占1个shared memory cache line。

## swizzle<2, 3, 3>的计算过程

swizzle<2, 3, 3>的计算过程如下：

`Bits=2, MBase=3, Shifts=3` 这样一个swizzle函数的计算过程：

1. **bits掩码**：根据`Bits`计算一个掩码：`bit_mask = (1 << Bits) - 1`，Bits是几，掩码就由几个1构成。例如，Bits=3，掩码为`111`。<font color="blue">Bits的长度决定了swizzle的二维空间中的列数为：$2^{\text{bits}}$。</font>
1. **yyy_mask和zzz_mask**：计算`yyy_mask`和`zzz_mask`。假设`MBase=3`，那么会有3个比特位保持不改变。从这3个比特位的尾部位置开始，向左数`Shifts`个bit位数是yyy_mask，向右数`Shifts`个bit位数是 <font color="blue">$\text{yyy\_mask}$和$\text{zzz\_mask}$决定了swizzle二维空间中要去交换的两个位置。</font>

<p align="center">
<img src="figures/swizzle_func.png" width=50%><br>
Fig. swizzle函数位运算示意
</p>

<p align="center">
<img src="figures/swizzled_offset.png" width=50%><br>
Fig. 原始的offset和swizzled offset之间的关系
</p>

3. **permute输入**：`offset ^ shiftr(offset & self.yyy_msk, self.shift)`

   1. offset的二进制表示与`yyy_mask`相与，右移`Shifts`位，结果记作offset1；<font color="blue">offset1是将offset中$\text{yyy\_mask}$对应位置的bit位保留原值，其余位置清零，然后取出来的部分移动到$\text{zzz\_mask}$所在的位置。</font>
   2. offset与offset1进行异或。<font color="blue">一个bit位与0异或结果不变，结果相当于offset中$\text{yyy\_mask}$对应的bit位offset中$\text{zzz\_mask}$对应的bit位进行异或，写入$\text{zzz\_mask}$对应的位置</font>

## 对16x16数据块进行swizzle

有一个16x16的数据块，进一步以$1 \times 8$为粒度被分成了$16 \times 2$个块（**这里我们先模糊行优先/列优先，具体对应到行优先/列优先时，只需要对这两个维度做相应的调整和适配**）。我们的目标是<font color=red>将这个$16 \times 16$的数据块以bank-conflict free的方式存储在shared memory中</font>。

这里我们考虑以下假设：

1. 将GPU的shared memory看作由8个bank构成，于是每个bank位宽128 bits，正好对应了上面提到的大小为$1\times 8$的一段数据；
1. 数据是以半精度存储，于是$1024/16=64$个半精度正好存储在一条shared memory cache line里面
1. 单线程访问128bit，<font color="red">于是8线程并发访存一次的数据恰好可以写入一整行shared memory cache line，而这一点是我们需要保证的</font>，这八个线程写入shared memory的bank id必须是0~8这个8个bank id的一个permutation，不可以落入同一个bank。

<p align="center">
<img src="figures/shared_memory_for_2-3-3.png" width=30%><br>
Fig. <2,3,3>设置下shared memory的逻辑编号
</p>

$<B=2,M=3,S=3>$这样一个swizzle函数会对：$2^2*2^3*2^3=4*8*8=16*16=256$个元素进行permute，也就是恰好对16x16的半精度进行permute。

下表是swizzle函数要去操作的bit位，红色位置的bit位进行异或（后三位总是当成是一个元素）。

|-|-|-|<font color="red">x</font>|<font color="red">x</font>|<font color=blue>-</font>|<font color=blue>-</font>|<font color=blue>-</font>|
|:--|:--|:--|:--|:--|:--|:--|:--|
|**<font color="red">x</font>**|**<font color="red">x<font>**|-|-|-|**<font color=blue>-</font>**|**<font color=blue>-</font>**|**<font color=blue>-</font>**|

Bits=2决定了bank id用2个bit位表示，也就是shared memory的8个bank被分成了两组。

|xor|00|01|10|11|
|:--|:--|:--|:--|:--|
|**00**|00|01|10|11|
|**01**|01|00|11|10|
|**10**|10|11|00|01|
|**11**|11|10|01|00|

从上面这个表是$B=2$时的异或表，可以看到异或具有封闭性；

swizzle的二维index空间中共有$2^2\times 2^3=4\times8=32$个坐标，我们在下表中表示中这个swizzle空间中所有index的十进制（上方）和对应的二进制（下方）：

|||||
|:--:|:--:|:--:|:--:|
|0<br>==00==0==00==|1<br>00001|2<br>00010|3<br>00011|
|4<br>00100|5<br>00101|6<br>00110|7<br>00111|
|8<br>01000|9<br>01001|10<br>01010|11<br>01011|
|12<br>01100|13<br>01101|14<br>01110|15<br>01111|
|16<br>10000|17<br>10001|18<br>10010|19<br>10011|
|20<br>10100|21<br>10101|22<br>10110|23<br>10111|
|24<br>11000|25<br>11001|26<br>11010|27<br>11011|
|28<br>11100|29<br>11101|30<br>11110|31<br>11111|

swizzled index（**下面的表格用红色和黑色将数据分成了两部分，可以看出来换序仅仅发生在同色的数据块之内**）：

|bank-id|0|1|2|3|4|5|6|7|
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
|**Access-0**|<font color=red>0</font>|<font color=red>1</font>|<font color=red>2</font>|<font color=red>3</font>|4|5|6|7|
|**Access-1**|<font color=red>9</font>|<font color=red>8</font>|<font color=red>11</font>|<font color=red>10</font>|13|12|15|14|
|**Access-2**|<font color=red>18</font>|<font color=red>19</font>|<font color=red>16</font>|<font color=red>17</font>|22|23|20|21|
|**Access-3**|<font color=red>27</font>|<font color=red>26</font>|<font color=red>25</font>|<font color=red>24</font>|31|30|29|28|

<p align="center">
<img src="figures/store_to_shared_memory.png"><br>
Fig. 以Global Memory上row major的16x16数据块为源，线程分数据方式以及shared memory中数据存储顺序；
</p>

# Reference

1. [What does bitwise XOR (exclusive OR) mean?](https://stackoverflow.com/questions/6398427/what-does-bitwise-xor-exclusive-or-mean)
1. [DEVELOPING CUDA KERNELS TO PUSH TENSOR CORES TO THE ABSOLUTE LIMIT ON NVIDIA A100](https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s21745-developing-cuda-kernels-to-push-tensor-cores-to-the-absolute-limit-on-nvidia-a100.pdf)
1. [cute 之 Swizzle](https://zhuanlan.zhihu.com/p/671419093)
