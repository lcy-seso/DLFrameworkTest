# Machines for Benchmark

## Device Information

||[RTX 2080 Ti](https://www.techpowerup.com/gpu-specs/geforce-rtx-2080-ti.c3305)|[A100 80G](https://www.techpowerup.com/gpu-specs/a100-pcie-80-gb.c3821)|[RTX A6000](https://www.techpowerup.com/gpu-specs/rtx-a6000.c3686)|
|:--|:--|:--|:--|
|Architecture|Turning|Ampere|Ampere|
|CUDA Capability Major/Minor version number|7.5|8.0|8.6|
|Total amount of global memory|10.75 GB|79.10 GB|44.55 GB|
|Multiprocessors, (064) CUDA Cores/MP|(068) Multiprocessors, (064) CUDA Cores/MP<br>**4352 CUDA Cores**|(108)Multiprocessors, (064) CUDA Cores/MP<br>**6912 CUDA Cores**|(084) Multiprocessors, (128) CUDA Cores/MP<br>**10752 CUDA Cores**|
|**<ins>GPU Max Clock rate</ins>**|1545 MHz (1.54 GHz)|1410 MHz (1.41 GHz)|1800 MHz (1.80 GHz)|
|**<ins>Memory Clock rate</ins>**|1750 Mhz|1512 Mhz|2000 MHz (2.00 GHz)|
|**<ins>Memory Type</ins>**|[GDDR6](https://en.wikipedia.org/wiki/GDDR6_SDRAM)|[HBM2e](https://en.wikipedia.org/wiki/High_Bandwidth_Memory)|[GDDR6](https://en.wikipedia.org/wiki/GDDR6_SDRAM)|
|**<ins>Memory Bus Width</ins>**|352-bit (44 bytes)|5120-bit|384-bit (48 bytes)|
|**Bandwidth (GB/s)**|616|1935|768|
|**<ins>Tensor Cores</ins>**|272|432|336|
|L2 Cache Size|5767168 bytes (5.5 MB)|41943040 bytes (40 MB)|6291456 bytes (6 MB)|
|Total amount of constant memory|65536 bytes (64 KB)|65536 bytes (64 KB)|65536 bytes (64 KB)|
|<ins>**Total amount of shared memory per block**</ins>|49152 bytes (48 KB)|49152 bytes (48 KB)|49152 bytes (48KB)|
|<ins>**Total shared memory per multiprocessor**</ins>|65536 bytes (64 KB)|167936 bytes (164 KB)|102400 bytes (100 KB)|
|Total number of registers available per block|65536|65536|65536|
|Warp size|32|32|32|
|Maximum number of threads per multiprocessor|1024|2048|1536|
|Maximum number of threads per block|1024|1024|1024|
|Max dimension size of a thread block (x,y,z)|(1024, 1024, 64)|(1024, 1024, 64)|(1024, 1024, 64)|
|Max dimension size of a grid size    (x,y,z)|(2147483647, 65535, 65535)|(2147483647, 65535, 65535)|(2147483647, 65535, 65535)
|Maximum memory pitch|2147483647 bytes|2147483647 bytes|2147483647 bytes|

## Bandwidth Test

Theoretical bandwidth of global memory is calculated as:

$$f_{mem} * \text{bus width} * DDR_{factor}$$

$f_{mem}$ is the memory frequency.

|Device|2080 Ti|A100 80G|A6000|
|:--|:--|:--|:--|
|$f_{mem}$ (MHz)|1750|1512|1800|
|bus width (bits)|352|5120|384|
|DDR factor|2 * 4|2 * 4|2 * 4|
|**Theoretical bandwidth (GB/s)**| **<ins>616</ins>** = 1750 * (352 / 8) * 8 / 1000 |**<ins>1935</ins>**| **<ins>768</ins>** = 2000 * (384 / 8) * 8 / 1000 |

PINNED Memory Transfers

|Test Name|2080 Ti|A100|A6000|
|--|:--|:--|:--|
|Host to Device (GB/s)|12.3|26.6|26.2|
|Device to Host (GB/s)|13.2|26.3|26.1|
|Device to Device (GB/s)|519.6|1321.1|568.6|
