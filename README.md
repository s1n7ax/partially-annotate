# partially-annotate
Partially annotate images for Computer Vision and generate project that can be
imported to the annotation tool

Image annotation for Computer Vision is a manual work and it's too painful. My
requirement is to partially train a `Yolo` neural network using bulk of data,
(manually annotated) and use that trained network to partially annotate
another bulk of data. User can review and re-correct what CNN got wrong.
As model improves, the number of correction user has to do is less.

I only had time to design it for my requirement. So this support reading from
`Darknet` only and generating `Supervise.ly` project type only.

## Prerequisites
* opencv (with CUDA support)
* python 3
* pip

## How to use
* Clone the project `git clone https://github.com/s1n7ax/partially-annotate.git`
* Create python env and activate
```
cd partially-annotator
python -m venv env
source env/bin/activate
```
* Install dependencies
```
pip install -r requirements.txt
```
* Add following `Yolo` files to `resources` directory
	* obj.cfg (`Yolo` configuration file used to train the it)
	* obj.weights (`Yolo` weights file)
	* obj.names (`Yolo` names files that contains all the classes)
* Move all the images, that needs to be partially annotated, to
  `resources/training` directory. (By default, this is picking all the \*.jpeg,
  \*.jpg, \*.png) files from `training` directory
* Run `python src/main.py` to generate the `Supervise.ly` project (blob size is 
set to 416 by default. You can change it in "DefaultNetworkFactory "class for DarknetYoloNetwork)
* Import the project "import plugin" to Supervisely when importing
* Open the project. You should have all the images partially annotated
* Now you can review and re-correct if needed

