# Handwriting Authorship Identification Using CNNs

## Motivation
This repository holds an attempt to apply transfer learning techniques and convolutional neural nets (CNNs) to model and predict the identity of the writer of a given handwritten image from the [CSAFE Handwriting Database](https://data.csafe.iastate.edu/HandwritingDatabase/?saveQueryContent=handwritingdbstudy-%3E++%28Writer_ID+%3C%3D+%270090%27%29+&files%5B%5D=&study=handwritingdbstudy&left-operands-parameters-name=Writer_ID&filter-operators-name=%3D&right-operands-parameters-value=Writer_ID&paramValues=0009#). This is intended to give insight into the potential applications of deep learning models in handwriting analysis with broad use in fields such as forensics, fraud detection, and historical document analysis.


## Overview
Handwriting author identification is important in forensic science, fraud detection, and document authentication. Traditional handwriting analysis depends on expert judgment, which can be subjective and time-consuming. This project explores whether deep-learning provides more consistent and scalable approaches to identifying writers from handwriting samples. Using the CSAFE Handwriting Database, which contains 475 individuals and 12,825 samples, author identification was framed as a multi-class image classification problem. Handwriting images were preprocessed and standardized to 384×384 formatting and divided into training, validation, and test sets. Convolutional neural networks with transfer learning were trained to classify samples. Initial 90-writer subsets achieved test accuracies between 72% and 78%. Larger-scale experiments using all data showed similar performance and achieved top-3 accuracies up to 90%. Because forensic applications require both accuracy and justification, this project emphasized explainable AI. Two explainability methods, Grad-CAM and LIME, were used to interpret predictions. These revealed that models sometimes relied on irrelevant features, such as white space and page margins, rather than handwriting characteristics, suggesting some predictions may not be based on trustworthy reasoning. To mitigate effects of possible confounding features, black & white color grading was applied to all images; models trained with this constraint achieved top-3 accuracies of 73% to 85%. These findings highlight the potential and limitations of AI-based handwriting identification. While deep-learning can improve efficiency, explainability is essential for ensuring systems are reliable for real-world applications.


## Summary of Work Done

### Data

* Source: [CSAFE Handwriting Database Version 5](https://data.csafe.iastate.edu/HandwritingDatabase/?saveQueryContent=handwritingdbstudy-%3E++%28Writer_ID+%3C%3D+%270090%27%29+&files%5B%5D=&study=handwritingdbstudy&left-operands-parameters-name=Writer_ID&filter-operators-name=%3D&right-operands-parameters-value=Writer_ID&paramValues=0009#)
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
 
### Data Preprocessing



### Exploratory Data Analysis (EDA)


### Modelling Approach

* Baseline model: started with EfficientNetB0
* Final moel: MobileNetV2
* Advanced models: attempted heavier architectures, e.g., ResNet50V2 & ConvNeXtTiny, though these did not yield comparative results

### Model Training

* Training hyperparameters:
  * Epochs: 70-100
  * Batch Size: 64
  * Image Size: 384×384
  * Learning Rate: default Adam optimizer (1e-3)
  * GPU: A100 GPU provided by Google Colab

## Results

Accuracy was the main metric chosen for evaluating models since all classes are equally represented and the multiclass nature of the problem made macro-precision or F1-scores less intuitive. Later models were also evaluated based on top-3 accuracy, which checks whether the correct prediction was represented within the top 3 guesses. This metric was chosen since it was more forgiving than raw accuracy, considering there are 475 different "options" to choose from.


### Model Interpretation


### Key Insights


### Conclusion


### Future Work

As, in its current form, the setup of the pipeline makes it a closed-set problem (i.e., writers outside of the training set cannot be predicted without additional appropriate training data), shifting focus from CNNs to Siamese Neural Networks (SNNs) is the next frame of focus. SNNs are designed to take two inputs and compare them against each other, and with all that was learned from the current pipeline regarding data cleaning and proper image handling specific to handwriting analysis, this would be the natural progression to testing the utility of deep learning and AI applications in digital forensics and other such spheres, and "opening" this problem (i.e., making it more generalizable). Other aspects to dive deeper into are more extreme augmentations and image processing steps. Augmentations have not been applied to any finalized models since, whenever executed, they worsened performance. However, applying different forms of augmentation (such as cropping to focus on words or letters rather than entire paragraphs) is yet to be tested. This would also serve to yield more data and inflate the training set.


## Repository Structure Explanation

### How to Run
**Generating final model**: Download data from the [CSAFE Handwriting Database](https://data.csafe.iastate.edu/HandwritingDatabase/?saveQueryContent=handwritingdbstudy-%3E++%28Writer_ID+%3C%3D+%270090%27%29+&files%5B%5D=&study=handwritingdbstudy&left-operands-parameters-name=Writer_ID&filter-operators-name=%3D&right-operands-parameters-value=Writer_ID&paramValues=0009#). Clone this repository, or download `BW_coloring.ipynb`. It is recommended to execute code virtually, such as with Google Colab, ensuring dependencies, such as `reload_data2.py`, are within the same/virtual directory. Run notebook and download .keras model and other output files; XAI interpretations are built into this notebook. Previous models can be trained and generated in the same way with their respective notebook(s).


### Contents of Repository
* **notebooks**: contains current code progress
  * **previous work**: subfolder containing initial modelling attempts; full documentation can be found in [this repository](https://github.com/zaisid/DATA4380_Vision)
* **data**: contains metadata, modules, and select model-specific test set data
* **images**: contains graphs and visualizations generated throughout pipeline, including loss/accuracy curves, test outputs, and Grad-CAM maps
* **models**: contains select models trained throughout different stages of the pipeline; majority are MobileNetV2 architectures
  * `AllClass5_bw.keras`: "final" model trained and tested on black & white color-graded images
  * `AllClass7_noPHR.keras`: updated version of `AllClass5_bw.keras` with smaller training/testing volume after removal of short prompts ("PHR") from dataset
  * `HighRes3.keras`: 90-class "final" model trained similarly as `AllClass5_bw.keras`
  * `MobileNetV2_age.keras`: early model where target variable was *age* rather than authorship
  * `MobileNetV2_hand.keras`: early model where target variable was *handedness* rather than authorship
  * `MobileNetV2_gender.keras`: early model where target variable was *gender* rather than authorship
  * `EfficientNet.keras`: baseline model from earliest stages of the project
* **presentations**: contains all presentations (i.e., slides and posters) made on this project
* **results**: contains loss and accuracy data over epochs for all trained models as .CSV files
* `requirements.txt`: lists required modules for deployment scripts


### Software Setup / Requirements
Google Colab was used for majority of model training for its computational processing resources. Visualizations were completed with matplotlib. Modelling and analysis was done through tensorflow, keras, numpy, and scikit-learn. Data organization was automated with the os, shutil, PIL, tqdm, and zipfile modules. Required and recommended modules are enumerated in `required.txt` as well as at the top of every notebook.


### Data Loading

The data can be downloaded from the [CSAFE Handwriting Database webpage](https://data.csafe.iastate.edu/HandwritingDatabase/?saveQueryContent=handwritingdbstudy-%3E++%28Writer_ID+%3C%3D+%270090%27%29+&files%5B%5D=&study=handwritingdbstudy&left-operands-parameters-name=Writer_ID&filter-operators-name=%3D&right-operands-parameters-value=Writer_ID&paramValues=0009#).


## Citations

Crawford, Amy; Ray, Anyesha; Carriquiry, Alicia; Kruse, James; Peterson, Marc (2019): CSAFE Handwriting Database. Iowa State University. Dataset. https://doi.org/10.25380/iastate.10062203.v1
