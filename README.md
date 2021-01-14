# CamVid Segmentation
## Deep Learning-based Semantic Segmentation for Autonomous Driving 

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

### Project Architecture

Please overview UML diagram that depicts major project components and dependencies:

![UML Component Diagram](/docs/uml_model.png)

The project have two executable scripts: 
* train.py
* evaluate.py

There is also several utility scripts with the following responsibilities:
* settings.py - add project related settings
* data.py - utility variables, functions, and classes responsible for accessing dataset images
* model.py - utility functions used to create segmentation model
* myresunet.py - utility functions used to create custom U-Net/ResNet inspired model
