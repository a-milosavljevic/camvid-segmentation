# CamVid Segmentation
## Deep Learning-based Semantic Segmentation for Autonomous Driving 

This project was developed as a part of the presentation that I gave on the 
[Programming 2.0 webinar: Autonomous driving](https://www.linkedin.com/events/programming2-0webinara-autonomo6755030263207665664/).

The presentation slides can be found here: 
[programming_2_0.pdf](./docs/programming_2_0.pdf)

### Instructions 

To be able to use the code please follow listed instructions:

1)  Download data from https://www.kaggle.com/carlolepelaars/camvid
 
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

### Project Overview

Please overview UML diagram that depicts major project components and dependencies:

![UML Component Diagram](./docs/uml_model.png)

The project have two executable scripts: 
* [train.py](./train.py) - used to train a model.
* [evaluate.py](./evaluate.py) - used to evaluate trained model against *test* set.

There is also several utility scripts with the following responsibilities:
* [settings.py](./settings.py) - add project related settings.
* [data.py](./data.py) - utility variables, functions, and classes responsible for accessing dataset images. 
  * Relies on *tensorflow* and *cv2* libs.
* [model.py](./model.py) - utility functions used to create segmentation model. 
  * Relies on *myresunet.py* script and *segmentation_models* lib.
* [myresunet.py](./myresunet.py) - utility functions used to create custom U-Net/ResNet inspired model.

For more info about *segmentation_models* lib please check its GitHub page:
https://github.com/qubvel/segmentation_models

### Experiments

A list of conducted experiments with appropriate accuracy achieved on the test set is reported in 
[experiments.csv](./results/experiments.csv).

The best accuracy of 90.05% (min. 70.86%, max. 97.40%, std. 5.68%) were achieved with U-Net model and EfficientNetB2 backbone. 
The trained model along with all other outputs for that experiment can be found 
[here](./results/unet_efficientnetb2_bs4/).
