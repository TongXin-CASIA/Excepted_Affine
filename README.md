# Excepted_Affine
Expected Affine: a registration method for damaged section in ssEM

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
in Windows 10 to create the environment.(Attention: opencv_world3412.dll is needed.)
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
