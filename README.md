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
pip install -r requerimentsWindows.txt
```

5) Let's play

If you have any problem with train yolov8 due to the settings.yaml, you have to open it and correct the datasets_dir to the yolov8 folder. Remember that you have to create a yolov5 and yolov8 folder to save the results of each model in his specific folder
