# test_task
Solution of test task

All code tested on python3.8

## Run Linux
    sudo pip3 install -r requirements.txt
    python3 main.py $folder_with_img
   
## Datasets
Trainig data was used from https://www.kaggle.com/jeffheaton/glasses-or-no-glasses
Validation data was used from example images
    
## Metrcis
<p align="center">
    <img src="roc_curve.png", width="480">
    <img src="precision_recall_curve.png", width="480">
    <img src="accuracy.png", width="480">
</p>


## Inference Optimization
To reduce time of model inference was used topology of efficientnet model with higly reduced number of filters in each layer
MTCNN was used as face detector
