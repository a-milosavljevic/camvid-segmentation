# CamVid Segmentation
##Deep Learning-based Semantic Segmentation for Autonomous Driving 

To be able to use the code please follow listed instructions:

1)  Download data from https://www.kaggle.com/carlolepelaars/camvid/download
 
2)  Extract files and place them into *data* folder using the following folder structure:
    ```    
    data/test/*.png
    data/test_labels/*.png
    data/train/*.png
    data/train_labels/*.png
    data/val/*.png
    data/val_labels/*.png
    data/class_dict.csv
    ```

4) Execute *train.py* to train the model based. The results will be placed in *tmp* folder. In case of an out of memory problem, adjust *batch size* in *settings.py*:  
   ```
   batch_size = 4
   ```
   
5) Execute *evaluate.py* to evaluate trained model on the test subset. The results will be placed in *tmp* folder.

6) Check *settings.py* for other training options.
