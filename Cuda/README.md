Some cuda tests.

# NVIDIA GeForce RTX 2080 Ti

## Device Information

|CUDA Capability Major/Minor version number|7.5|
|:--|:--|
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
|Maximum memory pitch:|2147483647 bytes|
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
|**Theoretical bandwidth (GB/s)**| 616 = 1750 * 352 / 8 * 8 / 1000 |

PINNED Memory Transfers

|Test Name|Bandwidth(GB/s)
|:--|:--|
|Host to Device|12.3|
|Device to Host|13.2|
|Device to Device|519.6|
