#downloads https://www.inf.ufpr.br/vri/databases/yj4Iu2-UFPR-ALPR.zip in /dataset and unzips it
import os
import zipfile

dataset_zip_path = "dataset/yj4Iu2-UFPR-ALPR.zip"
dataset_extract_path = "dataset/"

if not os.path.exists(dataset_extract_path):
    with zipfile.ZipFile(dataset_zip_path, "r") as zip_ref:
        zip_ref.extractall(dataset_extract_path)
