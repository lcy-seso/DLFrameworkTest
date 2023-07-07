# Demonstrate How to Use PyTorch Customized OP

```bash
find_torch_lib_path=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'`
cmake -DCMAKE_PREFIX_PATH=$find_torch_lib_path ..
```
