# [NVIDIA GeForce RTX 2080 Ti](https://www.techpowerup.com/gpu-specs/geforce-rtx-2080-ti.c3305)

## Device Information

|CUDA Capability Major/Minor version number|7.5|
|--|:--|
|Total amount of global memory|11012 MBytes (11546394624 bytes)|
|(068) Multiprocessors, (064) CUDA Cores/MP|4352 CUDA Cores|
|GPU Max Clock rate|1545 MHz (1.54 GHz)|
|Memory Clock rate|7000 Mhz|
|Memory Bus Width|352-bit (44 bytes)|
|L2 Cache Size|5767168 bytes (5.5 MB)|
|Total amount of constant memory|65536 bytes (64 MB)|
|Total amount of shared memory per block|49152 bytes (48 MB)|
|Total shared memory per multiprocessor|65536 bytes (64 MB)|
|Total number of registers available per block|65536 (64 MB)|
|Warp size|32|
|Maximum number of threads per multiprocessor|1024|
|Maximum number of threads per block|1024|
|Max dimension size of a thread block (x,y,z)|(1024, 1024, 64)|
|Max dimension size of a grid size    (x,y,z)|(2147483647, 65535, 65535)|
|Maximum memory pitch|2147483647 bytes|
|Texture alignment|12 bytes|

## Bandwidth Test

Theoretical bandwidth of global memory is calculated as:

$$f_{mem} * \text{bus width} * DDR_{factor}$$

$f_{mem}$ is the memory frequency.

|Device|2080 Ti|
|:--|:--|
|$f_{mem}$ (MHz)|1750|
|bus width (bits)|352|
|DDR factor|2 * 4|
|**Theoretical bandwidth (GB/s)**| 616 = 1750 * (352 / 8) * 8 / 1000 |

PINNED Memory Transfers

|Test Name|Bandwidth(GB/s)
|--|:--|
|Host to Device|12.3|
|Device to Host|13.2|
|Device to Device|519.6|

# [A100 80G PCIe](https://www.techpowerup.com/gpu-specs/a100-pcie-80-gb.c3821)

|CUDA Capability Major/Minor version number|8.0
|:--|:--|
|Total amount of global memory| 80995 MBytes (84929216512 bytes)|
|(108) Multiprocessors, (064) CUDA Cores/MP|6912 CUDA Cores|
|GPU Max Clock rate|1410 MHz (1.41 GHz)|
|Memory Clock rate|1512 Mhz|
|Memory Bus Width|5120-bit|
|L2 Cache Size|41943040 bytes|
|Total amount of constant memory|65536 bytes|
|Total amount of shared memory per block|49152 bytes|
|Total shared memory per multiprocessor|167936 bytes|
|Total number of registers available per block| 65536|
|Warp size|32|
|Maximum number of threads per multiprocessor|2048|
|Maximum number of threads per block|1024|
|Max dimension size of a thread block (x,y,z)|(1024, 1024, 64)|
|Max dimension size of a grid size    (x,y,z)|(2147483647, 65535, 65535)|
|Maximum memory pitch|2147483647 bytes|
|Concurrent copy and kernel execution|Yes with 8 copy engine(s)|

## Bandwidth Test

1555 GB/s (?)

|Device|A100|
|:--|:--|
|$f_{mem}$ (MHz)|1512|
|bus width (bits)|5120|
|DDR factor|2 * 4|
|**Theoretical bandwidth (GB/s)**| ? = 1512 * (5120 / 8) * 8 / 1000 |

PINNED Memory Transfers
|Test Name|Bandwidth(GB/s)
|:--|:--|
|Host to Device|26.6|
|Device to Host|26.3|
|Device to Device|1321.1|
