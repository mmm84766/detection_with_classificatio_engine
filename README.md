# Object Detection and Classification


## Object Detection

Object detection takes a captured image as input and produces the bounding boxes as the output to be fed into the classification model. We use TensorFlow Object Detection API, which is an open source framework built on top of TensorFlow to construct, train and deploy object detection models. The Object Detection API also comes with a collection of detection models pre-trained on the COCO dataset that are well suited for fast prototyping. Specifically, we use a lightweight model: ```ssd_mobilenet_v1_coco``` that is based on Single Shot Multibox Detection (SSD) framework with minimal modification.  Though this is a general-purpose detection model (not optimized specifically for object detection), we find this model sufficiently met our needs, achieving the balance between good bounding box accuracy (as shown in the following figure) and fast running time.

![img](readme_imgs/examples_detection.png)

###Object Classification

After locating the bounding box for the object, we crop the image to only include the head, resize it to 32x32, and pass this along to the object classification step.


We use a simple CNN for our classification. It consists of three convolutional layer with (3x3 kernel), the last two of which are followed by a max_pooling layer, a flatten layer, and two fully connected layers. 


## Implementation 

### Objects Detection

For the classes included in COCO dataset, please see pascal\_label\_map.pbtxt, genarally located in 
```
${PYTHON_DIRECTORY}$/site-packages/tensorflow/models/object_detection/data
```

The object detection is implemented in 
```
get_localization(self, image, visual=False) 
```
member function of the TLClassifier class, in tl\_detection\_classification_test.py. The following boiler plate code are used to initialize tensorflow graph and model

```
detect_model_name = 'ssd_mobilenet_v1_coco_11_06_2017'
PATH_TO_CKPT = detect_model_name + '/frozen_inference_graph.pb'
# setup tensorflow graph
self.detection_graph = tf.Graph()
    
# configuration for possible GPU use
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# load frozen tensorflow detection model and initialize 
# the tensorflow graph
with self.detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
       serialized_graph = fid.read()
       od_graph_def.ParseFromString(serialized_graph)
       tf.import_graph_def(od_graph_def, name='')
       self.sess = tf.Session(graph=self.detection_graph, config=config)
       self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
       # Each box represents a part of the image where a particular object was detected.
        self.boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
       # Each score represent how level of confidence for each of the objects.
       # Score is shown on the result image, together with the class label.
        self.scores =self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections =self.detection_graph.get_tensor_by_name('num_detections:0')

```
The actual detection is carrried out by 

```
(boxes, scores, classes, num_detections) = self.sess.run(
                  [self.boxes, self.scores, self.classes, self.num_detections],
                  feed_dict={self.image_tensor: image_expanded})
```

There could be multiple detections of object in an image, here, I select the first occurence of the detection, i.e., the one with the highest confidence.
```
idx = next((i for i, v in enumerate(cls) if v == 1), None)
```
To avoid false-positive, I select a confidence threshold and reject any detection that is lower than this threshold. In case that this threshold by itself can not prevent false-positive in some corner cases (as shown in the figure below), I also set the box size and height-to-width ratio thresholds.



### Object Classification
The training and evaluation of the classifier is implemented in /classification/tl_classification.py
The CNN architecture for the classifier is shown in the following:

 <img src="readme_imgs/model.png" alt="Drawing" style="width: 220px;"/>
 
 
We save the model into the .h5 file to be used in the two-stage approach.


### Requirements
Use tensorflow 1.15.0 or previous version.



### How to run code and get final result
First you need to paste some picture in images folder. then run the file

 ```
    tl_detection_classification_test.py
    
 
 ```
 
 After successful run the script, you will get your final result in same terminal where you will get location of objects and class of objects.
 
 
