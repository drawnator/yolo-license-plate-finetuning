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

dataset_url = "https://github.com/drawnator/yolo-license-plate-finetuning/releases/download/plate_dataset/Brazil.Plates.Detector.v2i.yolo26.zip"
dataset_zip_path = "datasets/brazil_yolo12.zip"
dataset_extract_path = "datasets/brazil_yolo12"

face_dataset_url = "https://github.com/drawnator/yolo-license-plate-finetuning/releases/download/face_dataset/FACE.DETECTION.FYP.v1i.yolov12.zip"
face_dataset_zip_path = "datasets/face_yolo12.zip"
face_dataset_extract_path = "datasets/face_yolo12"

ALPR_dataset_url = "https://www.inf.ufpr.br/vri/databases/yj4Iu2-UFPR-ALPR.zip"
ALPR_dataset_zip_path = "datasets/yj4Iu2-UFPR-ALPR.zip"
ALPR_dataset_extract_path = "datasets/"
# The ALPR zip extracts into this folder; used to detect if it's already prepared.
ALPR_dataset_final_path = "datasets/UFPR-ALPR dataset"

os.makedirs("datasets", exist_ok=True)


def prepare_dataset(url, zip_path, extract_path, final_path=None):
    """Download and extract a dataset, skipping work that's already done.

    - If `final_path` (or `extract_path` when no `final_path` is given) already
      exists, do nothing.
    - If the zip is already present, skip the download and just extract.
    - Otherwise download then extract. The zip is removed only after a
      successful extraction.
    """
    presence_path = final_path or extract_path
    if os.path.exists(presence_path) and os.path.isdir(presence_path) and os.listdir(presence_path):
        print(f"[skip] {presence_path} already exists")
        return

    if not os.path.exists(zip_path):
        print(f"[download] {url} -> {zip_path}")
        subprocess.run(["curl", "-L", url, "-o", zip_path], check=True)
    else:
        print(f"[skip download] {zip_path} already exists")

    print(f"[extract] {zip_path} -> {extract_path}")
    subprocess.run(["unzip", "-q", zip_path, "-d", extract_path], check=True)
    subprocess.run(["rm", zip_path], check=True)


prepare_dataset(dataset_url, dataset_zip_path, dataset_extract_path)
prepare_dataset(face_dataset_url, face_dataset_zip_path, face_dataset_extract_path)
prepare_dataset(
    ALPR_dataset_url,
    ALPR_dataset_zip_path,
    ALPR_dataset_extract_path,
    final_path=ALPR_dataset_final_path,
)

from pathlib import Path
import shutil
import os
dimensions = [1_920,1_080]
original_path = "./datasets/UFPR-ALPR dataset"
folder_path = './datasets/UFPR-ALPR'

def get_data(lines):
    id_type = lines[2].split(":")[-1].strip()
    vehicle_pos = list(map(int,lines[1].split(":")[-1].strip().split()))
    plate_pos = lines[7].split(":")[-1].strip().split()
    plate_pos = [list(map(int,i.split(","))) for i in plate_pos]
    return id_type, vehicle_pos, plate_pos 

def convert_vehicle(vehicle_pos):
    x = vehicle_pos[0]/dimensions[0]
    y = vehicle_pos[1]/dimensions[1]
    w = vehicle_pos[2]/dimensions[0]
    h = vehicle_pos[3]/dimensions[1]
    return [x+w/2,y+h/2,w,h]

def convert_plate(plate_pos):
    p1 = plate_pos[0]
    p2 = plate_pos[1]
    p3 = plate_pos[2]
    p4 = plate_pos[3]
    xs = [p1[0],p2[0],p3[0],p4[0]]
    ys = [p1[1],p2[1],p3[1],p4[1]]
    x = sum(xs)/4 /dimensions[0]
    y = sum(ys)/4 /dimensions[1]
    w = (max(xs)-min(xs)) /dimensions[0]
    h = (max(ys)-min(ys)) /dimensions[1]
    return [x,y,w,h]

for root, dirs, files in os.walk(original_path):
    for file in files:
        if file.endswith(".txt"):
            if file == "README.txt": continue
            path = os.path.join(root, file)
            new_path = Path(path.replace(original_path,folder_path).replace("ALPR","ALPR/labels"))
            new_path.parent.mkdir(parents=True, exist_ok=True)
            id_type, vehicle_pos, plate_pos = "","",""
            with open(path) as f:
                lines = f.readlines()
                id_type, vehicle_pos, plate_pos = get_data(lines)
                if id_type == "car":id_type = 2
                else: id_type = 3
                vehicle_pos = convert_vehicle(vehicle_pos)
                plate_pos = convert_plate(plate_pos)
                # print(id_type,vehicle_pos,plate_pos)
            with open(new_path,"w") as f:
                f.write(f"0 {plate_pos[0]} {plate_pos[1]} {plate_pos[2]} {plate_pos[3]}\n")
                f.write(f"{id_type} {vehicle_pos[0]} {vehicle_pos[1]} {vehicle_pos[2]} {vehicle_pos[3]}")
            # print(new_path)
        else:
            path = os.path.join(root, file)
            new_path = Path(path.replace(original_path,folder_path).replace("ALPR","ALPR/images"))
            new_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(new_path))
        