# Excepted_Affine
This repository contains the official implementation of the paper
"[Expected affine: A registration method for damaged section in serial sections electron microscopy](https://doi.org/10.3389/fninf.2022.944050)"

![Python 3.9](https://img.shields.io/badge/python-3.9-green.svg?style=plastic) ![PyTorch 1.10.0](https://img.shields.io/badge/pytorch-1.10.0-green.svg?style=plastic) 
![OpenCV 3.4](https://img.shields.io/badge/opencv-3.4-green?style=plastic)![Qt 5](https://img.shields.io/badge/Qt-5-green?style=plastic)
![license GNU](https://img.shields.io/github/license/TongXin-CASIA/Excepted_Affine?style=plastic)
## Using the Code
### Requirements
This code has been developed under Python 3.9, PyTorch 1.10, OpenCV 3.4, Qt5 and Ubuntu 16.04 or Windows 10.
If the link library cannot be found, please check whether opencv and QT are set correctly.

In addition to the above libraries, the python environment can be set as follows:

```shell
conda create -n Exception_Affine python=3.9
conda activate Exception_Affine
pip3 install opencv-python torch
pip3 install scipy pillow scikit-image matplotlib
```
or you can use 
```shell
conda env create -f .\ExceptAffine.yml
```
in Windows 10 to create the environment.(Attention: opencv_world3412.dll is needed, and it can be extracted from opencv_world3412.zip under the root directory.)
### Register two sections
```Register
python reg.py -f DATA/DOLW7_0.png -m DATA/DOLW7_1.png -ma DATA/DOLW7_m.png -o output.png
```

### Run the Registration Experiment

    python reg_exp.py -f img_fixed_path -m img_moving_path -ma img_mask_path -o out_path

### Datasets in the paper
https://github.com/TongXin-CASIA/Damaged_Section

### Label the defect
You can choose to label folds or cracks manually, or use the following tools to label automatically.

https://github.com/jabae/detectEM

Note that the quality of automatic annotation may be affected by samples.

### Citation

If you think this work is useful for your research, please consider citing:

```
@ARTICLE{10.3389/fninf.2022.944050,
  
AUTHOR={Xin, Tong and Shen, Lijun and Li, Linlin and Chen, Xi and Han, Hua},   
	 
TITLE={Expected affine: A registration method for damaged section in serial sections electron microscopy},      
	
JOURNAL={Frontiers in Neuroinformatics},      
	
VOLUME={16},           
	
YEAR={2022},      
	  
URL={https://www.frontiersin.org/articles/10.3389/fninf.2022.944050},       
	
DOI={10.3389/fninf.2022.944050},      
	
ISSN={1662-5196},   
   
ABSTRACT={Registration is essential for the volume reconstruction of biological tissues using serial section electron microscope (ssEM) images. However, due to environmental disturbance in section preparation, damage in long serial sections is inevitable. It is difficult to register the damaged sections with the common serial section registration method, creating significant challenges in subsequent neuron tracking and reconstruction. This paper proposes a general registration method that can be used to register damaged sections. This method first extracts the key points and descriptors of the sections to be registered and matches them via a mutual nearest neighbor matcher. K-means and Random Sample Consensus (RANSAC) are used to cluster the key points and approximate the local affine matrices of those clusters. Then, K-nearest neighbor (KNN) is used to estimate the probability density of each cluster and calculate the expected affine matrix for each coordinate point. In clustering and probability density calculations, instead of the Euclidean distance, the path distance is used to measure the correlation between sampling points. The experimental results on real test images show that this method solves the problem of registering damaged sections and contributes to the 3D reconstruction of electronic microscopic images of biological tissues. The code of this paper is available at <ext-link ext-link-type="uri" xlink:href="https://github.com/TongXin-CASIA/Excepted_Affine" xmlns:xlink="http://www.w3.org/1999/xlink">https://github.com/TongXin-CASIA/Excepted_Affine</ext-link>.}
}
``
