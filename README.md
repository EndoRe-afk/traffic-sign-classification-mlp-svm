

# Traffic Sign Classification with MLP and SVM
A machine learning project comparing a Multilayer Perceptron (MLP) and Support Vector Machine (SVM) for multiclass traffic sign recognition across 43 traffic-sign classes.
The project focuses on building a fair and reproducible comparison between both models using the same preprocessing pipeline, group-aware data splitting, hyperparameter tuning, and final evaluation on an untouched test set.
Key Results
Model	Accuracy	Macro Precision	Macro Recall	Macro F1
MLP	0.5149	0.6333	0.7092	0.5752
SVM	0.4467	0.5293	0.6795	0.5037


The final MLP achieved approximately 6.82 percentage points higher accuracy and 7.15 percentage points higher macro F1 than the selected SVM.
Dataset
The project uses the public Traffic Signs Classification dataset from Kaggle.
- 73,139 original image files
- 43 traffic-sign classes
- 11,819 exact duplicate files removed using SHA-256 hashing
- 61,320 unique image files retained
- related frames from the same traffic-sign sequence were grouped to reduce data leakage
Preprocessing
Each image was processed using the same pipeline for both models:
1. convert images to three RGB channels
2. resize images to 32 × 32
3. normalise pixel intensities from 0–255 to 0–1
4. flatten each image into a 3,072-element feature vector
5. keep related frames from the same traffic-sign sequence in the same dataset split
The nominal group split was:
- 64% training
- 16% validation
- 20% testing
Because sequence groups contain different numbers of images, the final image counts were:
- 32,010 training images
- 12,960 validation images
- 16,350 test images
After model selection, the training and validation sets were combined to give 44,970 images for final model fitting.
MLP Model
The MLP experiments investigated:
- hidden-layer width
- learning rate
- training convergence
Tested configurations included:
Hidden Neurons	Learning Rate	Iterations	Training Loss	Validation Accuracy	Validation Macro F1
64	0.003	79	0.217	0.547	0.588
128	0.003	89	0.121	0.531	0.584
128	0.001	152	0.036	0.518	0.579


The final MLP used:
- one hidden layer with 64 neurons
- ReLU activation
- Adam optimiser
- learning rate 0.003
- batch size 128
- maximum of 200 iterations
The final model converged after 97 iterations.
SVM Model
Three SVM kernels were investigated:
- linear
- radial basis function
- polynomial
Each kernel was tested using:
- C = 0.1
- C = 1
- C = 10
For RBF and polynomial kernels, gamma='scale' was used.
Stage 1
All nine predefined SVM configurations were screened using:
- 30% of the initial training groups
- 10,770 images
- 204 groups
- 3-fold StratifiedGroupKFold
- macro F1 as the model-selection metric
The three highest-ranked candidates all used a linear kernel.
Stage 2
The two strongest configurations were retrained using the full initial training partition.
Configuration	Validation Accuracy	Macro Precision	Macro Recall	Macro F1
Linear SVM, C=0.1	0.4911	0.5231	0.6700	0.4898
Linear SVM, C=1	0.4959	0.5166	0.6792	0.4912


The final SVM used a linear kernel with C=1.
Evaluation
The final models were evaluated once using the untouched test set.
Metrics included:
- accuracy
- macro precision
- macro recall
- macro F1-score
- per-class precision, recall and F1
- row-normalised confusion matrices
Macro-averaged metrics were used because class sizes were uneven and each traffic-sign class should contribute equally to the comparison.
Key Findings
- the MLP achieved stronger overall test performance than the SVM under the tested configurations
- lower MLP training loss did not necessarily lead to better validation performance
- the SVM linear kernel performed better than the tested RBF and polynomial configurations during screening
- several visually similar traffic-sign classes were frequently confused
- flattened RGB pixels provide a simple shared representation but do not explicitly preserve the two-dimensional spatial structure of the image
Examples of class-level confusion included:
- Speed limit 60 km/h frequently predicted as Speed limit 80 km/h
- Keep right frequently predicted as Go straight or right
Project Structure
.
├── COMPSYS306_Project_1_FINAL.py
├── archive.zip
├── README.md
└── outputs_staged_full_dataset_local/
    ├── test_comparison_corrected.csv
    ├── per_class_model_comparison_corrected.csv
    ├── mlp_confusion_matrix_normalized.png
    ├── svm_confusion_matrix_normalized.png
    ├── final_mlp_model.pkl
    ├── final_svm_model.pkl
    └── run_config_final.json
Generated output filenames may vary slightly depending on the final script version.
Requirements
Main Python libraries:
numpy
pandas
matplotlib
scikit-learn
scikit-image
Install dependencies with:
pip install numpy pandas matplotlib scikit-learn scikit-image
Running the Project
Place the original dataset archive beside the Python script:
archive.zip
COMPSYS306_Project_1_FINAL.py
Then run:
python COMPSYS306_Project_1_FINAL.py
The script will perform preprocessing, duplicate removal, dataset splitting, hyperparameter tuning, final model fitting, evaluation, and output generation.
Limitations
Both models use flattened RGB pixel values as their input representation. This removes explicit spatial relationships between neighbouring pixels, which can make visually similar signs harder to distinguish.
Hyperparameter exploration was also intentionally limited to keep computational cost manageable. The selected models should therefore be interpreted as the best configurations among those tested rather than globally optimal models.
Future Work
Possible extensions include:
- convolutional neural networks
- image augmentation
- feature extraction methods that preserve local image structure
- broader hyperparameter optimisation
- dimensionality reduction
- comparison against additional classical and deep-learning classifiers
Technologies
- Python
- NumPy
- Pandas
- scikit-learn
- scikit-image
- Matplotlib
Author
Joshua Kao
Computer Systems Engineering
University of Auckland
