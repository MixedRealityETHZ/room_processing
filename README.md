# Room processing

The room processing code of "3D Room Arrangements using Virtual Reality" project.

The semantic segmentation part is based on the Minkowski Engine

## Minkowski Engine

The Minkowski Engine is an auto-differentiation library for sparse tensors. It supports all standard neural network layers such as convolution, pooling, unpooling, and broadcasting operations for sparse tensors. For more information, please visit [the documentation page](http://nvidia.github.io/MinkowskiEngine/overview.html).

### Anaconda

```
# create conda env

conda update -n base -c defaults conda
conda install openblas-devel -c anaconda

# install pytorch conda

git clone https://github.com/NVIDIA/MinkowskiEngine.git

cd MinkowskiEngine

python setup.py install

conda install -c conda-forge pyembree

pip install open3d
```

