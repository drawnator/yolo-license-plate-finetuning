#downloads https://www.inf.ufpr.br/vri/databases/yj4Iu2-UFPR-ALPR.zip in /dataset and unzips it
import subprocess
import os 
# dataset_zip_path = "dataset/yj4Iu2-UFPR-ALPR.zip"
# dataset_extract_path = "dataset/"

# if not os.path.exists(dataset_extract_path):
#     with zipfile.ZipFile(dataset_zip_path, "r") as zip_ref:
#         zip_ref.extractall(dataset_extract_path)


# downloades https://universe.roboflow.com/ds/FfrIWBIMBq?key=7CTRoFmULV in /dataset and unzips it into dataset/brazil_yolo12
import requests
import zipfile

dataset_url = "https://github.com/drawnator/yolo-license-plate-finetuning/releases/download/dataset/Brazil.Plates.Detector.v2i.yolo26.zip"
dataset_zip_path = "app/datasets/brazil_yolo12.zip"
dataset_extract_path = "app/datasets/brazil_yolo12"

#curl -L "https://universe.roboflow.com/ds/FfrIWBIMBq?key=7CTRoFmULV" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
if not os.path.exists(dataset_zip_path):
    subprocess.run(["curl","-L",dataset_url,"-o",dataset_zip_path])
if not os.path.exists(dataset_extract_path):
    subprocess.run(['unzip', dataset_zip_path,"-d", dataset_extract_path])
    subprocess.run(['rm', dataset_zip_path])
