# conda create a new environment
# make sure system cuda version is the same with pytorch cuda
# follow the instruction of Pyotrch Geometric: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
export PATH=/usr/local/cuda-10.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64:$LD_LIBRARY_PATH

conda create --name amps_env 
# activate this enviroment
conda activate amps_env

conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
# test if pytorch is installed successfully
python -c "import torch; print(torch.__version__)"
nvcc --version # should be same with that of torch_version_cuda (they should be the same)
python -c "import torch; print(torch.version.cuda)"

pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu116.html

pip install tqdm
pip install ogb
### check the version of ogb installed, if it is not the latest
python -c "import ogb; print(ogb.__version__)"
# please update the version by running
#pip install -U ogb
conda install -c conda-forge rdkit
#Install modlamp for metadata
pip install modlamp


