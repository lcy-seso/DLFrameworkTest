# Device Information for My Tests

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
|<ins>**Maximum number of threads per multiprocessor**</ins>|1024<br>(**32 warps**)|2048<br>(**64 warps**)|1536<br>(**48 warps**)|
|Maximum number of threads per block|1024|1024|1024|
|Max dimension size of a thread block (x,y,z)|(1024, 1024, 64)|(1024, 1024, 64)|(1024, 1024, 64)|
|Max dimension size of a grid size    (x,y,z)|(2147483647, 65535, 65535)|(2147483647, 65535, 65535)|(2147483647, 65535, 65535)
|Maximum memory pitch|2147483647 bytes|2147483647 bytes|2147483647 bytes|

# Bandwidth Test

Theoretical bandwidth of global memory is calculated as:

$$f_{mem} * \text{bus width} * DDR_{factor}$$

$f_{mem}$ is the memory frequency.

|Device|2080 Ti|
|:--|:--|
|$f_{mem}$ (MHz)|1750|
|bus width (bits)|352|
|DDR factor|2 * 4|
|**Theoretical bandwidth (GB/s)**| 616 = 1750 * 352 / 8 * 8 / 1000 |

PINNED Memory Transfers

|Test Name|Bandwidth(GB/s)
|:--|:--|
|Host to Device|12.3|
|Device to Host|13.2|
|Device to Device|519.6|
