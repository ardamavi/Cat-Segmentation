# Cat-Segmentation
### By Arda Mavi

Cat segmentation with deep learning.<br/>
Database created by myself.

### Segmentation Example:
|<img src="/Data/Train_Data/input/cat.78.jpg" width="200">|<img src="Data/Train_Data/mask/mask_cat.78.jpg" width="200">|
|:-:|:-:|
| Orijinal | Segmented |


### Using Predict Command:
`python3 predict.py <ImageFileName>`

### Model Training:
`python3 train.py`

### Using TensorBoard:
`tensorboard --logdir=Data/Checkpoints/logs`

### Model Architecture:
- Input Data
Shape: 64x64x3

- Convolutional Layer
32 filter
Filter shape: 3x3
Strides: 1x1

- Activation
Function: ReLu

- Convolutional Layer
64 filter
Filter shape: 3x3
Strides: 1x1

- Activation
Function: ReLu

- Transpose Convolutional Layer
64 filter
Filter shape: 3x3
Strides: 1x1

- Activation
Function: ReLu

- Merge Layer

- Transpose Convolutional Layer
1 filter
Filter shape: 3x3
Strides: 1x1

- Activation
Function: Sigmoid

##### Optimizer: Adadelta
##### Loss: Dice Coefficient

### Important Notes:
- Used Python Version: 3.6.0

- Install necessary modules with `sudo pip3 install -r requirements.txt` command.

- We work on 64x64 image also if you use bigger, program will automatically return to 64x64.
