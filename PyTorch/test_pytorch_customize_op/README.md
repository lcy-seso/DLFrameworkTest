# Demonstrate How to Use PyTorch Customized OP

```bash
cache_file=CMakeCache.txt
if test -f "$cache_file"; then
    rm "$cache_file"
fi

cache_dir=CMakeFiles
if [ -d "$cache_dir" ]; then
    rm -rf "$cache_dir"
fi

find_torch_lib_path=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'`
cmake -DCMAKE_PREFIX_PATH=$find_torch_lib_path ..
```
