# BME450-Identify_Brain_Tumors_in_MRIs
    "Identifying Different Types of Tumors in Brain MRIs"
## Team Members
    Mei Pitts (meipitts), Nathan Petrucci (npetrucci)
## Project Description
    The main goal of this project will be processing a large dataset of MRI images already labeled as either being normal or containing a specific kind of brain tumor.
    The resulting trained neural net will be capable of determining if an inputted MRI image is that of a patient with a glioma, meningioma, pituitary tumor or no tumor.
    The relative strength of the output (how close the each 'tumor' neuron output is to 1 compared to how close the 'normal' neuron is to 1 will also serve as a measure for of the model's confidence in its prediction. 
    We will make use of the existing dataset titled "Brain Tumor MRI Dataset" which is available at https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data?select=Testing. 
    The dataset is separated into training and test portions. 
    Images will be scaled down to pixel dimensions that are more easily processed on a laptop computer, different dimensions may be experimented with to determine what dimensions yield the best results while keeping computation time within reasonable limits. 
    The training portion of the dataset will be used to train the neural net, and the neural net will subsequently be tested using the test dataset. The code will follow the basic form of the code used for HW01 and other in-class examples
