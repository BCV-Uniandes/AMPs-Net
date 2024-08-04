# Use NVIDIA PyTorch image as the base
FROM nvcr.io/nvidia/pytorch:22.03-py3

# Base pytorch 
RUN conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.6 -c pytorch -c conda-forge

# Set required versions for each core dependency using cu116
RUN pip install torch-scatter==2.0.9 torch-sparse==0.6.14 torch-cluster==1.6.0 torch-spline-conv==1.2.1 torch-geometric==2.1.0 -f https://data.pyg.org/whl/torch-1.12.0+cu116.html

# Copy the requirements.txt file into the container
COPY requirements.txt .
RUN pip install -r requirements.txt

# Set the default command to bash
CMD ["/bin/bash"]

