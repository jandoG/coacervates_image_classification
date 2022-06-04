## Notes from explorations 

#### 1. A simple CNN with keras 
- Read images from "classname" folders
- Populate arrays - train and validation data 
- Classname - from the folder path 
- Train a simple CNN network  
- Prediction 

**Issues:**

- Data too small, we have < 800 images per class and > 50k parameters 

#### 2. Find trained models and do transfer learning 
- There are many models available in literature which have been already trained for image classification problems, for e.g. RESNET50, ImageNet, VGG 
- These models+ their weights could be easily imported using Keras and applied to image classification problems. 
- The ImageNet is trained to classify from 1k classes common object classes like dog, cat, table, chair etc. 

Most useful blogpost:
https://machinelearningmastery.com/how-to-use-transfer-learning-when-developing-convolutional-neural-network-models/ 

**Issues:**

- These models were trained on RGB images and the images we have are greyscale 
- A very interesting discussion here on using these networks on greyscale images: https://stackoverflow.com/questions/51995977/how-can-i-use-a-pre-trained-neural-network-with-grayscale-images 
- Suggested workarounds:
	1. Replicate the greyscale into 3 channels 
		- All 3 channels will have same info  
		- Use the model as is and modify the last layer (softmax) for 3 classes 
	2. Change the base layer to input 1 channel greyscale image and modify the weights after the first layer 
		- this changes the architecture; also since the color information might be used in these networks in the first layers, using grayscale might make the further layers not so efficient. 
- Something I found for greyscale: https://gist.github.com/mohapatras/5d720cb19014ed573bcd3ed36c5929f5

### Most promising approach with existing tools and time constraints:
*Used in final workflow.*

- **Scikit-image**: works best with feature detection + random forest classification 
- features: **texture, blob** like features work best for the data 
