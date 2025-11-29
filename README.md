# Handwriting Authorship Identification Using CNNs

This repository holds an attempt to apply transfer learning techniques and convolutional neural nets (CNNs) to model and predict the identity of the writer of a given handwritten image from the [CSAFE Handwriting Databas](https://data.csafe.iastate.edu/HandwritingDatabase/?saveQueryContent=handwritingdbstudy-%3E++%28Writer_ID+%3C%3D+%270090%27%29+&files%5B%5D=&study=handwritingdbstudy&left-operands-parameters-name=Writer_ID&filter-operators-name=%3D&right-operands-parameters-value=Writer_ID&paramValues=0009#). 



## Summary of Work Done

### Data

* Type: Image data, scans of handwriting samples; .csv file containing metadata on each writer is also included when downloading the data.
* Size: 
* Instances: a 60/20/20 train/validation/test split was used, with stratified sampling to ensure all classes were equally represented in each set


### Contents of Repository
* **notebooks**: contains current code progress
  * **previous work**: subfolder containing initial modelling attempts; full documentation can be found in [this repository](https://github.com/zaisid/DATA4380_Vision)


### Software Setup
Google Colab was used for majority of model training for its computational processing resources. Visualizations were completed with matplotlib. Modelling and analysis was done through tensorflow, keras, numpy, and scikit-learn. Data organization was automated with the os, shutil, and zipfile modules. 


### Data

The data can be downloaded on the [CSAFE Handwriting Database webpage](https://data.csafe.iastate.edu/HandwritingDatabase/?saveQueryContent=handwritingdbstudy-%3E++%28Writer_ID+%3C%3D+%270090%27%29+&files%5B%5D=&study=handwritingdbstudy&left-operands-parameters-name=Writer_ID&filter-operators-name=%3D&right-operands-parameters-value=Writer_ID&paramValues=0009#).


## Citations

Crawford, Amy; Ray, Anyesha; Carriquiry, Alicia; Kruse, James; Peterson, Marc (2019): CSAFE Handwriting Database. Iowa State University. Dataset. https://doi.org/10.25380/iastate.10062203.v1
