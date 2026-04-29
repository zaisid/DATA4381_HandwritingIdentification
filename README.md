# Handwriting Authorship Identification Using CNNs

This repository holds an attempt to apply transfer learning techniques and convolutional neural nets (CNNs) to model and predict the identity of the writer of a given handwritten image from the [CSAFE Handwriting Databas](https://data.csafe.iastate.edu/HandwritingDatabase/?saveQueryContent=handwritingdbstudy-%3E++%28Writer_ID+%3C%3D+%270090%27%29+&files%5B%5D=&study=handwritingdbstudy&left-operands-parameters-name=Writer_ID&filter-operators-name=%3D&right-operands-parameters-value=Writer_ID&paramValues=0009#). 



## Summary of Work Done

### Data

* Type: image data, scans of handwriting samples; .CSV file containing metadata on each writer is also included when downloading the data.
* Size: approx. 5GB 
* Instances: 12,825 images
  * 475 classes (authors), 27 images each
  * Each class is organized as such:
    * 3 different, conserved prompts (LND, WOZ, PHR) written 9 times each by each author
    * the 9 instances of each prompt were written across 3 different sessions
    * 3 repetitions of each prompt were made per session
    * all above aspects are described by image's file name; e.g., `w0028_s03_pWOZ_r02.png` is writer **w0028**'s second repetition of the **WOZ** prompt in the third session
  * a 15/6/6 (approx. 60/20/20 ratio) train/validation/test split count was used on majority of models with stratified sampling to ensure all classes were equally represented in each set


### Contents of Repository
* **notebooks**: contains current code progress
  * **previous work**: subfolder containing initial modelling attempts; full documentation can be found in [this repository](https://github.com/zaisid/DATA4380_Vision)
* **data**: contains metadata, modules, and select model-specific test set data
* **images**: contains graphs and visualizations generated throughout pipeline, including loss/accuracy curves and Grad-CAM maps
* **models**: contains select models trained throughout different stages of the pipeline; majority are MobileNetV2 architectures
  * `AllClass5_bw.keras`: "final" model trained and tested on black & white color-graded images
  * `AllClass7_noPHR.keras`: updated version of `AllClass5_bw.keras` with smaller training/testing volume after removal of short prompts ("PHR") from dataset
  * `HighRes3.keras`: 90-class "final" model trained similarly as `AllClass5_bw.keras`
  * demographic models
    * `MobileNetV2_age.keras`: early model where target variable was *age* rather than authorship
    * `MobileNetV2_hand.keras`: early model where target variable was *handedness* rather than authorship
    * `MobileNetV2_gender.keras`: early model where target variable was *gender* rather than authorship
  * `EfficientNet.keras`: baseline model from earliest stages of the project
* **presentations**: contains all presentations (i.e., slides and posters) made on this project
* **results**: contains loss and accuracy data over epochs for all trained models as .CSV files
* `requirements.txt`: lists recommended modules for executing pipeline


### Software Setup
Google Colab was used for majority of model training for its computational processing resources. Visualizations were completed with matplotlib. Modelling and analysis was done through tensorflow, keras, numpy, and scikit-learn. Data organization was automated with the os, shutil, PIL, tqdm, and zipfile modules. Required and recommended modules are enumerated in `required.txt`.


### Data

The data can be downloaded on the [CSAFE Handwriting Database webpage](https://data.csafe.iastate.edu/HandwritingDatabase/?saveQueryContent=handwritingdbstudy-%3E++%28Writer_ID+%3C%3D+%270090%27%29+&files%5B%5D=&study=handwritingdbstudy&left-operands-parameters-name=Writer_ID&filter-operators-name=%3D&right-operands-parameters-value=Writer_ID&paramValues=0009#).


## Citations

Crawford, Amy; Ray, Anyesha; Carriquiry, Alicia; Kruse, James; Peterson, Marc (2019): CSAFE Handwriting Database. Iowa State University. Dataset. https://doi.org/10.25380/iastate.10062203.v1
