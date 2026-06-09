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

# copyparty ?zip downloads: the zip contains the folder's contents directly
# (no wrapping top-level directory), so we extract into the final target folder.

dataset_url = "https://copyparty.guilherme.zip/share/Brazil.Plates.Detector.v2i.yolo26?zip"
dataset_zip_path = "datasets/brazil_yolo12.zip"
dataset_extract_path = "datasets/brazil_yolo12"

face_dataset_url = "https://copyparty.guilherme.zip/share/FACE.DETECTION.FYP.v1i.yolov12?zip"
face_dataset_zip_path = "datasets/face_yolo12.zip"
face_dataset_extract_path = "datasets/face_yolo12"

ALPR_dataset_url = "https://copyparty.guilherme.zip/share/UFPR-ALPR%20dataset?zip"
ALPR_dataset_zip_path = "datasets/UFPR-ALPR-dataset.zip"
ALPR_dataset_extract_path = "datasets/UFPR-ALPR"

RODOSOL_dataset_url = "https://copyparty.guilherme.zip/share/RodoSol-ALPR?zip"
RODOSOL_dataset_zip_path = "datasets/RodoSol-ALPR.zip"
RODOSOL_dataset_extract_path = "datasets/RodoSol-ALPR"
os.makedirs("datasets", exist_ok=True)


def prepare_dataset(url, zip_path, extract_path):
    """Download and extract a dataset, skipping work that's already done.

    - If `extract_path` already exists and is non-empty, do nothing.
    - If the zip is already present, skip the download and just extract.
    - Otherwise download then extract. The zip is removed only after a
      successful extraction.
    """
    if os.path.exists(extract_path) and os.path.isdir(extract_path) and os.listdir(extract_path):
        print(f"[skip] {extract_path} already exists")
        return

    if not os.path.exists(zip_path):
        print(f"[download] {url} -> {zip_path}")
        subprocess.run(["curl", "-L", url, "-o", zip_path], check=True)
    else:
        print(f"[skip download] {zip_path} already exists")

    os.makedirs(extract_path, exist_ok=True)
    print(f"[extract] {zip_path} -> {extract_path}")
    subprocess.run(["unzip", "-q", zip_path, "-d", extract_path], check=True)
    subprocess.run(["rm", zip_path], check=True)


prepare_dataset(dataset_url, dataset_zip_path, dataset_extract_path)
prepare_dataset(face_dataset_url, face_dataset_zip_path, face_dataset_extract_path)
prepare_dataset(ALPR_dataset_url, ALPR_dataset_zip_path, ALPR_dataset_extract_path)
prepare_dataset(RODOSOL_dataset_url, RODOSOL_dataset_zip_path, RODOSOL_dataset_extract_path)

from pathlib import Path
import shutil
import os

## converts UFPR ALPR dataset to yolov5 format

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

## converts RODOSOL dataset to yolo format
#
# RodoSol-ALPR annotation files have this shape:
#   type: car            (or motorcycle)
#   plate: ODE2510
#   layout: Brazilian    (or Mercosur)
#   corners: x1,y1 x2,y2 x3,y3 x4,y4
#
# Images are 1280x720. The dataset only provides plate corners (no vehicle
# bounding box), so we emit a single YOLO label per image: the plate (class 0).
# split.txt assigns each image to training / validation / testing.

rodosol_dimensions = [1_280, 720]
rodosol_original_path = "./datasets/RodoSol-ALPR"
rodosol_folder_path = "./datasets/RodoSol-ALPR"


def _parse_rodosol_label(lines):
    """Parse a RodoSol annotation file's lines into (vehicle_type, corners)."""
    vehicle_type = ""
    corners = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip()
        if key == "type":
            vehicle_type = value.lower()
        elif key == "corners":
            corners = [list(map(int, p.split(","))) for p in value.split()]
    return vehicle_type, corners


def _corners_to_yolo_bbox(corners, dims):
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    w = (x_max - x_min) / dims[0]
    h = (y_max - y_min) / dims[1]
    cx = (x_min + x_max) / 2 / dims[0]
    cy = (y_min + y_max) / 2 / dims[1]
    return [cx, cy, w, h]


_RODOSOL_SPLIT_MAP = {
    "training": "training",
    "train": "training",
    "validation": "validation",
    "valid": "validation",
    "val": "validation",
    "testing": "testing",
    "test": "testing",
}


def convert_rodosol():
    split_file = os.path.join(rodosol_original_path, "split.txt")
    if not os.path.exists(split_file):
        print(f"[skip rodosol] {split_file} not found")
        return

    images_root = Path(rodosol_folder_path) / "images"
    labels_root = Path(rodosol_folder_path) / "labels"
    # Skip if the split-based structure already exists (training/ subfolder).
    # The raw extracted images live under images/cars-br/ etc. so we can't just
    # check for *.jpg existence — we need the reorganized layout.
    if labels_root.exists() and any(labels_root.rglob("*.txt")):
        print(f"[skip rodosol] {rodosol_folder_path} already prepared")
        return

    with open(split_file) as f:
        entries = [line.strip() for line in f if line.strip()]

    converted = 0
    skipped = 0
    for entry in entries:
        rel_image, _, split_name = entry.partition(";")
        split_name = _RODOSOL_SPLIT_MAP.get(split_name.strip().lower())
        if not split_name:
            skipped += 1
            continue

        # rel_image looks like "./images/cars-br/img_000003.jpg"
        rel_image = rel_image.lstrip("./")
        src_image = Path(rodosol_original_path) / rel_image
        src_label = src_image.with_suffix(".txt")
        if not src_image.exists() or not src_label.exists():
            skipped += 1
            continue

        # Preserve the subgroup folder (cars-br, cars-me, ...) under the split.
        subgroup = src_image.parent.name
        image_name = src_image.name
        label_name = src_label.with_suffix(".txt").name

        dst_image = images_root / split_name / subgroup / image_name
        dst_label = labels_root / split_name / subgroup / label_name
        dst_image.parent.mkdir(parents=True, exist_ok=True)
        dst_label.parent.mkdir(parents=True, exist_ok=True)

        with open(src_label) as fh:
            vehicle_type, corners = _parse_rodosol_label(fh.readlines())
        if len(corners) != 4:
            skipped += 1
            continue

        plate_bbox = _corners_to_yolo_bbox(corners, rodosol_dimensions)

        # Class 0 = plate. Vehicle bbox is not provided by RodoSol so we only
        # emit the plate label. Vehicle type is preserved in a comment for
        # debugging / future use.
        with open(dst_label, "w") as fh:
            fh.write(
                f"0 {plate_bbox[0]} {plate_bbox[1]} {plate_bbox[2]} {plate_bbox[3]}\n"
            )

        shutil.copy2(str(src_image), str(dst_image))
        converted += 1

    print(f"[rodosol] converted={converted} skipped={skipped}")


convert_rodosol()
