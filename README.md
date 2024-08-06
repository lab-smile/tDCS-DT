# Towards tDCS Digital Twins using Deep Learning-based Direct Estimation of Personalized Electrical Field Maps from T1-Weighted MRI

Transcranial Direct Current Stimulation (tDCS) is a non-invasive brain stimulation method that applies neuromodulatory effects to the brain via low-intensity, direct current. It has shown possible pos- itive effects in areas such as depression, substance use disorder, anxiety, and pain. Unfortunately, mixed trial results have delayed the field’s progress. Electrical current field approximation provides a way for tDCS researchers to estimate how an individual will respond to specific tDCS parameters. Publicly available physics-based stimulators have led to much progress; however, they can be error-prone, susceptible to quality issues (e.g., poor segmentation), and take multiple hours to run. Digital functional twins provide a method of estimating brain function in response to stimuli using computational methods. We seek to implement this idea for individualized tDCS. Hence, this work provides a proof-of-concept for generating electrical field maps for tDCS directly from T1-weighted magnetic resonance images (MRIs). Our deep learning method employs special loss regularizations to improve the model’s generalizability and calibration across individual scans and electrode montages. Users may enter a desired electrode montage in addition to the unique MRI for a custom output. Our dataset includes 442 unique individual heads from individuals across the adult lifespan. The pipeline can generate results on the scale of minutes, unlike physics-based systems that can take 1-3 hours. Overall, our methods will help streamline the process of individual current dose estimations for improved tDCS interventions.

## Paper
This repository provides the official implementation of training and evaluation of the model as described in the following paper:

**Towards tDCS Digital Twins using Deep Learning-based Direct Estimation of Personalized Electrical Field Maps from T1-Weighted MRI**

Skylar E. Stolte<sup>1</sup>, Aprinda Indahlastari<sup>2,3</sup>, Alejandro Albizu<sup>2,4</sup>, Adam J. Woods<sup>2,3,4</sup>, and Ruogu Fang<sup>1,2,5*</sup>

<sup>1</sup> J. Crayton Pruitt Family Department of Biomedical Engineering, Herbert Wertheim College of Engineering, University of Florida (UF), USA<br>
<sup>2</sup> Center for Cognitive Aging and Memory, McKnight Brain Institute, UF, USA<br>
<sup>3</sup> Department of Clinical and Health Psychology, College of Public Health andHealth Professions, UF, USA<br>
<sup>4</sup> Department of Neuroscience, College of Medicine, UF, USA<br>
<sup>5</sup> Department of Electrical and Computer Engineering, Herbert Wertheim College ofEngineering, UF, USA<br>

International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI) 2024<br>
[paper](TO BE ADDED) | [code]https://github.com/lab-smile/tDCS-DT | [poster](TO BE ADDED)

## Major results from our work

- Custom dataset of T1-weighted MRIs and tDCS current maps for 442 individual heads across the lifespan
- Dual training system and loss regularization that also incorporates calibration
- Proof-of-concept results for the generation of tDCS current estimations directly from MRIs using deep learning

<div align="center">
	<img src = "Images/Figure1-1533.pdf">
</div>

<div align="center">
  <b>fig. 1:</b> Method Pipeline.<br>
</div>
<br>

## Usage
You can find there are two MATLAB codes, you can directly change the directory to your own data. You need to select the DOMINO++ working folder and add to path before you running these two MATLAB codes. 

In case of you are using different version of MATLAB, if you are using MATLAB 2020b, you need to change line 56 to :
```
image(index) = tissue_cond_updated.Labels(k)
```
Then you can run the combine_mask.m. The output should be a Data folder with the following structure: 
```
Data ImagesTr sub-TrX_T1.nii sub-TrXX_T1.nii ... 
ImagesTs sub-TsX_T1.nii sub-TsXX_T1.nii ...
LabelsTr sub-TrX_seg.nii sub-TrXX_seg.nii ...
LabelsTs sub-TsX_seg.nii sub-TsX_seg.nii ...
```
Maneuver to the /your_data/Data/. Run make_datalist_json.m

After this code is done, you may exit MATLAB and open the terminal to run the other codes.

### Build container
The DOMINO++ code uses the MONAI, an open-source foundation. We provide a .sh script to help you to build your own container for running your code.

Run the following code in the terminal, you need to change the line after --sandbox to your desired writable directory and change the line after --nv to your own directory.
```
sbatch building_container_v110.sh
```

The output should be a folder named monaicore110 under your desired directory.

### Training
Once the data and the container are ready, you can train the model by using the following command:
```
sbatch train.sh
```
Before you training the model, you need to make sure change the following directory:
- change the first singularity exec -nv to the directory includes monaicore110, for example: /user/DOMINOPlusPlus/monaicore110
- change the line after --bind to the directory includes monaicore110
- change the data_dir to your data directory
- change the model name to your desired model name
You can also specify the max iteration number for training. For the iterations = 100, the training progress might take about one hours, and for the iterations = 25,000, the training progress might take about 24 hours. 

### Testing
The test progress is very similar to the training progress. You need to change all paths and make sure the model_save_name matches your model name in runMONAI.sh. Then running the runMONAI_test.sh with the following command: 
```
sbatch test.sh
```
The outputs for each test subject is saved as a mat file.

### Pre-trained models
You can also use the pre-trained models we provide for testing, please fill out the following request form before accessing the DOMINO++ models.
Download pre-trained models [here](https://forms.gle/3GPnXXvWgaM6RZvr5)


## Acknowledgement

This work was supported by the National Institutes ofHealth/National Institute on Aging (NIA RF1AG071469, NIA R01AG054077),the National Science Foundation (1908299), and the NSF-AFRL INTERN Supplement (2130885). 


We employ UNETR as our base model from:
https://github.com/Project-MONAI/research-contributions/tree/main/UNETR

## Contact
For any discussion, suggestions and questions please contact: [Skylar Stolte](mailto:skylastolte4444@ufl.edu), [Dr. Ruogu Fang](mailto:ruogu.fang@bme.ufl.edu).

*Smart Medical Informatics Learning & Evaluation Laboratory, Dept. of Biomedical Engineering, University of Florida*