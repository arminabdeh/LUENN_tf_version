# LUENN

The development of Single-Molecule Localization Microscopy (__SMLM__) has enabled the visualization of sub-cellular structures, but its temporal resolution is limited. To address this issue, a deep-convolutional neural network called __LUENN__ has been introduced, which uses a unique architecture that rejects the isolated emitter assumption. LUENN is a Python package based on a deep CNN that utilizes the [Tensorflow](http://tensorflow.org/) tool for SMLM. It is capable of achieving high accuracy for a wide range of imaging modalities and frame densities. <br>

## 3D reconstruction
3D reconstruction of a live cell using LUENN <br>

https://user-images.githubusercontent.com/61014265/219693582-acd024b2-b547-496d-9136-95d91459288e.mp4

## System Requirements
The software was tested on a Linux system with Ubuntu version 7.0, and a Windows system with Windows 10 Home.
Training and evaluation were run on a standard workstation equipped with 32 GB of memory, an Intel(R) Core(TM) i7 − 8700, 3.20 GHz CPU, and a NVidia GeForce Titan Xp GPU with 12 GB of video memory.
 
# Installation
1. Download this repository as a zip file (or clone it using git). <br>
2. Go to the downloaded directory and unzip it. <br>
3. The conda environment for this project is given in environment_<os>.yml where <os> should be substituted with your operating system. For example, to replicate the environment on a linux system use the command: conda env create -f environment_linux.yml from within the downloaded directory. This should take a couple of minutes. <br>
4. After activation of the environment using: conda activate LUENN, you're set to go!

## Contributers:

__Armin Abdehkakha__, _Email: arminabd@buffalo.edu_<br>
__Craig Snoeyink__, _Email: craigsno@buffalo.edu_

