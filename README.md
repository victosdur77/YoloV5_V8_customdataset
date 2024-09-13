# YoloV5_V8_customdataset

1) Create an environment with Python 3.9 version (it is important that it is with conda):

```bash
conda create -n entorno python=3.9
```

2) activate the enviromment

```bash
conda actívate entorno
```

3) Clone the data reduction repository: SurveyGreenAI and follow the installation steps (https://github.com/Cimagroup/SurveyGreenAI.git). The final installation step takes a few minutes.

```bash
git clone https://github.com/Cimagroup/SurveyGreenAI.git
```

4) Install python dependecies needed

```bash
pip install -r requeriments.txt
```

5) Let's play

If you have any problem with train yolov8 due to the settings.yaml, you have to open it and correct the datasets_dir to the yolov8 folder. Remember that you have to create a yolov5 and yolov8 folder to save the results of each model in his specific folder.

In order to perform the experiments on the Roboflow dataset, you must use yolov5GAM24_roboflow.ipynb or yolov8GAM24_roboflow.ipynb, where you can choose the reduction method, as well as the reduction percentage.

In order to perform the experiments on the Mobility Aid dataset, you must use yolov5GAM24_mobilityaid.ipynb or yolov8GAM24_mobilityaid, where you can choose the reduction method, as well as the reduction percentage.

To make it work, you must download the following files that you can find at http://mobility-aids.informatik.uni-freiburg.de/ and save them in the DatasetMobilityAid folder.

- RGB images
- Annotations RGB
- Annotations RGB test set 2
- image set textfiles
- 
Once downloaded, you must go to the DataFormatYolov5.ipynb file inside the DatasetMobilityAid folder and run it completely, so that these images are in the proper YoloV5 format, and you can now use their specific notebooks.
