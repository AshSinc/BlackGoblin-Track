# BlackGoblin-Track

## **Abstract**

This repository is the result of a short investigation into classification of footwear and ground type, combined with object tracking for the purpose of automatated audio dubbing of footsteps in video.

The problem can be split into 3 main categories. Footwear classification, ground type classification, and object tracking of individual feet.

## **Footwear Classification**

To build a classification model to differentiate footwear types, we need a labelled dataset of footwear. We found "Fashion MNIST", "Shoe Dataset", and "UT Zappos50k".

### **Dataset**

**Fashion MNIST** - https://www.kaggle.com/datasets/zalando-research/fashionmnist?resource=download&select=fashion-mnist_train.csv <br>
Fashion MNIST consists of 70k images of 28x28 greyscale. Although only a small subset of these are footwear, and only sneaker and boots are labelled.

**Shoe Dataset** - https://www.kaggle.com/datasets/noobyogi0100/shoe-dataset <br>
Shoe Dataset consists of 249 images categorised by Boots, Sneakers, Flip flops, Loafers, Sandals, and Football boots. This would likely be too small in practice.

**UT Zappos50k** - https://vision.cs.utexas.edu/projects/finegrained/utzap50k/ <br>
UT Zappos50k dataset seemed the most promising consisting of 50k images categorised by Shoes, Sandal, Slipper, Boot. These categories are further subdivided into sub-types.

![This is an image](resources/images/utzap.png)<br>UT Zappos Dataset examples

### **Classifier**

Because YOLO can perform fast object tracking and identification, we experimented with transfer learning with yolov5 small and large networks using pretrained weights. We provided the model with train test split of UT Zappos50k data converted to YOLO format. It quickly became aparent that the model would not transfer well to real world examples, likely due to the identical orientation and lighting of each shoe in the dataset, and for some the fact that no feet were inside of them (strappy heels and sandals). 

We attempted to resolve this by augmenting the dataset in a number of ways such as changing background colour, skewing or rotating images and greyscaling the image before passing for training. However in the vast majority of cases YOLO could not find the shoe, even in an image filled with shoes, and when it did it would frequently misidentify.

### **Generating more data with Pose estimation dataset**

Another idea was to augment our existing Zappos dataset by providing it with annotated "Shoes in the wild" images. As it would be very time consuming to collect a significant number and annotate manually we opted to scrape the data from a pose estimation dataset, MPII Human Pose Dataset.

**MPII Human Pose Dataset** - http://human-pose.mpi-inf.mpg.de/

The Pose directory in this repo contains the python scripts required to convert from their matlab structure into a more understandable JSON structure with readmat_output_json.py. And then generate_dataset_from_json.py will take this JSON structure along with the dataset directory, and copy images where both ankles are visible in the frame. We draw bounding boxes around the expected foot location based on the annotated scale of the individual and their ankle position.

<img src="resources/images/023956189.jpg" width="500" /><br>Good annotation with MPII Pose. This could work  well for augmentation.

<img src="resources/images/062329813.jpg" width="500" /><br>Obscured annotation with MPII Pose. This does not work and would require manual removal from the dataset.

So there are two problems here. The dataset does not tell us if the feet themselves are obscured. If this was a statistically insignificant number it may not matter, but of the first 100 images roughly 40% were obscured in some way. Another issue is that if we are to use these generated images to augment the existing training data, we would still need to label the shoe type manually. This means that manually labelling is looking more like a necessity for accurate shoe classification.

............................Ideally we would have panoramic views of every shoe type in varying lighting conditions and then train a classifier on these types. But the dataset does not yet exist.

## **Ground Type Classification**

### **Dataset**

### **Classifier**

## **Object Tracking**

### **Model**