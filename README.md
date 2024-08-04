# Rational Discovery of Antimicrobial Peptides by Means of Artificial Intelligence

Paola Ruiz Puentes, Maria C. Henao, Javier Cifuentes, Carolina Muñoz-Camargo, Luis H. Reyes, Juan C. Cruz  and Pablo Arbeláez.

This repository contains the official implementation of AMPs-Net: [Rational Discovery of Antimicrobial Peptides by Means of Artificial Intelligence](https://www.mdpi.com/2077-0375/12/7/708/htm). 

## Paper

[Rational Discovery of Antimicrobial Peptides by Means of Artificial Intelligence](https://www.mdpi.com/2077-0375/12/7/708/htm).<br/>
[Paola Ruiz Puentes](https://paolaruizp.github.io)<sup>1,2</sup>, [Maria C. Henao](https://www.researchgate.net/profile/Maria-Henao-18)<sup>3</sup>, [Javier Cifuentes](https://scholar.google.com/citations?user=JpVIbNsAAAAJ&hl=es&oi=ao)<sup>2</sup>, [Carolina Muñoz-Camargo](https://scholar.google.com/citations?user=dOIitb4AAAAJ&hl=es&oi=ao)<sup>2</sup>,[Luis H. Reyes](https://scholar.google.com/citations?user=2vO8IrIAAAAJ&hl=es&oi=ao)<sup>3</sup>, [Juan C. Cruz](https://scholar.google.com/citations?user=k--wE0YAAAAJ&hl=es&oi=ao)<sup>2</sup>, [Pablo Arbeláez](https://scholar.google.com.co/citations?user=k0nZO90AAAAJ&hl=en)<sup>1</sup><br/>
Membranes MDPI, 2022.<br><br>

<sup>1 </sup> Center  for  Research  and  Formation  in  Artificial  Intelligence .([CINFONIA](https://cinfonia.uniandes.edu.co/)),  Universidad  de  los  Andes,  Bogotá 111711, Colombia. <br/>
<sup>2 </sup> Department  of  Biomedical  Engineering,  Universidad  de  los  Andes,  Bogotá 111711, Colombia.<br/>
<sup>3 </sup> Grupo de Diseño de Productos y Procesos (GDPP), Department of Chemical and Food Engineering, Universidad de los Andes, Bogota 111711, Colombia.<br/>

## Installation

The following steps are required in order to run AMPs-Net using a local Conda environment:<br />

```bash
$ export PATH=/usr/local/cuda-11.0/bin:$PATH 
$ export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH 

$ conda create --name amps_env 
$ conda activate amps_env 

$ bash amps_env.sh
```

Then you should be able to run the inference using the binary classification model:
 ```bash
./run_inference_AMP.sh  \ 
    -m <model_inference_path> \ 
    -b <batch_size> \ 
    -d <device> \ 
    -n <num_metadata> \ 
    -f <file_name_inference>
```

or using the multilabel model:

 ```bash
./run_inference_multilabel.sh \ 
    -m <model_inference_path> \ 
    -b <batch_size> \ 
    -d <device> \ 
    -n <num_metadata> \ 
    -f <file_name_inference>
```

Command parameters:

- **model_inference_path**: is the checkpoints model path used for inference, either `checkpoints/BINARY_AMPs` or `checkpoints/MULTILABEL_AMPs`.
- **batch_size**:  how many samples per batch to load in the torch DataLoader
- **device**: GPU device number to use, usually 0.
- **num_metadata**: number of metadata features for the graph representation

## Installation with Docker

You will need a machine with a CUDA compatible GPU with the NVIDIA Container Toolkit installed.

Please follow these steps:

1.  Install [Docker](https://www.docker.com/).
    *   Install
        [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
        for GPU support.
    *   Setup running
        [Docker as a non-root user](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user).

2.  Clone this repository and `cd` into it.

    ```bash
    git clone https://github.com/BCV-Uniandes/AMPs-Net
    cd AMPs-Net
    ```

3. Prepare your inference files in the `Inference` folder, see the [Usage](#Usage) section.

4. Build the Docker image:

    ```bash
    docker build -t amp-net:latest .
    ```

5. Run inference with the binary classification model. e.g, using the provided `Example.csv` file:

    ```bash
    docker run \
        -it --rm --gpus all \
        -v "$(pwd)":/workspace \
        amp-net:latest \
        bash -c "/workspace/run_inference_AMP.sh -m checkpoints/BINARY_AMPs -b 4 -d 0 -n 8 -f Example.csv"
    ```

6. Run inference with the multilabel model. e.g, using the provided `Example.csv` file:

    ```bash
    docker run \
        -it --rm --gpus all \
        -v "$(pwd)":/workspace \
        amp-net:latest \
        bash -c "/workspace/run_inference_multilabel.sh -m checkpoints/MULTILABEL_AMPs -b 4 -d 0 -n 8 -f Example.csv"
    ```

7. Check the predictions in the `Inference` folder.

## Models
We provide pretrained models available for download in the following [link](http://157.253.243.19/AMPs-Net/).
Last update on the models on the 01/01/2023.

## Usage
To train each of the components of our method: please refer to run_AMP.sh and run_multilabel.sh.

To perform inference on your data please refer to: run_inference_AMP.sh and run_inference_multilabel.sh.

**To setup your data on the proper format for our models please refer to generate_metadata.sh. Your dataset should be a csv file with a column name Sequence and your peptides in their AA sequence. Follow the example in data/datasets/Inference.**

To perform inference with your models please modify L34-37 from inference.py. (Change Checkpoint.pth to Checkpoint__valid_best.pth)

## Citation

We hope you find our paper useful. To cite us, please use the following BibTeX entry:

```
@article{RuizPuentes2022,
  doi = {10.3390/membranes12070708},
  url = {https://doi.org/10.3390/membranes12070708},
  year = {2022},
  month = jul,
  publisher = {{MDPI} {AG}},
  volume = {12},
  number = {7},
  pages = {708},
  author = {Paola Ruiz Puentes and Maria C. Henao and Javier Cifuentes and Carolina Mu{\~{n}}oz-Camargo and Luis H. Reyes and Juan C. Cruz and Pablo Arbel{\'{a}}ez},
  title = {Rational Discovery of Antimicrobial Peptides by Means of Artificial Intelligence},
  journal = {Membranes}
}
```

