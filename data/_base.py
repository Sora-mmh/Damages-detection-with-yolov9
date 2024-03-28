import json
import logging
from pathlib import Path
import shutil
from typing import List

import cv2
import numpy as np
from PIL import ImageColor

logging.basicConfig(level=logging.INFO)

class DatasetFormatter():
    
    # Members 
    _data_pth : Path
    _new_imgs_pth : Path 
    _labels_pth : Path 
    _root_dir : Path 

    def __init__(self,
                 data_pth : Path,
                 root_dir: Path ) -> None:
        self._data_pth = data_pth
        self._root_dir = root_dir
        self._metadata_pth = data_pth / "meta.json"
        self._annots_pth = data_pth / "File1" / "ann"
        self._imgs_pth = data_pth / "File1" / "img"
        self._new_imgs_pth = root_dir / "base"
        self._imgs_dir = root_dir / "images"
        self._labels_dir = root_dir / "labels"
        logging.info("Start extracting classes and rectifying their indices ...")
        self.get_cls_with_indices_from_metadata()
        logging.info("Extraction and rectification are completed.")
        logging.info("Start loading images and annotations ...")
        self.get_imgs_with_annots()
        logging.info("Loading images and annotations are completed.")
        self._new_formatted_cls_indices = {k:i for i,(k,_) in enumerate(self._cls_indices.items())}
        logging.info("Start organising directory for images and annotations and correct files names ...")
        self.move_imgs_and_correct_files_names()
        logging.info("Correction and organisation are completed.")
        logging.info("Start splitting data ...")
        self.split_data()
        logging.info("Splitting data is completed.")
        # self.rectify_pths() # to adapt paths to your directory (eg: on drive for colab importation)


    def get_cls_with_indices_from_metadata(self):
        with open(self._metadata_pth) as f:
            classes = json.load(f)
        self._cls_indices = {cls['title']: (cls['id'], cls['color']) for cls in classes['classes']}

    def get_imgs_with_annots(self):
        self._imgs_annots = []
        for pth in self._annots_pth.iterdir():
            img_annot = {pth.stem : []}
            with open(pth) as f:
                objects = json.load(f)["objects"]
            for obj in objects:
                img_annot[pth.stem].append([obj["classId"], obj["classTitle"], obj["points"]["exterior"]])
            self._imgs_annots.append(img_annot)
    
    def move_imgs_and_correct_files_names(self):
        self.stems = []
        for img_annot in self._imgs_annots:
            (img, annots), = img_annot.items()
            img_pth = (imgs_pth / img).as_posix()
            stem = img[:-4].split(" ")[-1]
            self.stems.append(stem)
            new_img_name = stem + ".jpg"
            annot_name = stem + ".txt"
            new_img_pth = (self._new_imgs_pth /  new_img_name).as_posix()
            shutil.copy(img_pth, new_img_pth)
            img_arr = cv2.imread(new_img_pth)
            if img_arr is not None:
                img_h, img_w, _ = img_arr.shape
                with open((self._labels_dir /  annot_name).as_posix(), "w") as f:
                    for annot in annots:
                        #polygon_data = " ".join(str(e) for e in flatten_matrix(annot[2]))
                        x_c, y_c, w, h = convert_polygon_to_bbox(annot[2])
                        #print("bbox : ", x_c, y_c, w, h, img_w, img_h)
                        normalized_bbox = [round(x_c / img_w, 2), round(y_c / img_h, 2), round(w / img_w, 2), round(h / img_h, 2)]
                        #print("normalized bbox :", normalized_bbox)
                        bbox_data = " ".join(str(e) for e in normalized_bbox)
                        row_data = str(self._new_formatted_cls_indices[annot[1]]) + " " + bbox_data
                        f.write(row_data + "\n")

    def split_data(self, split_ratio : tuple = (0.7, 0.2, 0.1)):
        #all_imgs_pths = [pth for pth in self._new_imgs_pth.iterdir() if pth.as_posix().endswith(".jpg")]
        #all_annots_pths = [annots_pth for annots_pth in self._labels_dir.iterdir() if annots_pth.as_posix().endswith(".txt")] 
        all_imgs_pths = [self._new_imgs_pth / (stem + ".jpg")  for stem in self.stems]
        all_annots_pths = [self._labels_dir / (stem + ".txt")  for stem in self.stems]

        imgs_pths = {
                    "train" : all_imgs_pths[:int(len(all_imgs_pths)*split_ratio[0])],
                    "val" : all_imgs_pths[int(len(all_imgs_pths)*split_ratio[0]):int(len(all_imgs_pths)*(1 - split_ratio[2]))], 
                    "test" : all_imgs_pths[int(len(all_imgs_pths)*(1 - split_ratio[2])):] 
                    }
        annots_pths = {
                    "train" : all_annots_pths[:int(len(all_annots_pths)*split_ratio[0])],
                    "val" : all_annots_pths[int(len(all_annots_pths)*split_ratio[0]):int(len(all_annots_pths)*(1 - split_ratio[2]))],
                    "test" : all_annots_pths[int(len(all_imgs_pths)*(1 - split_ratio[2])):]
                    }
        for data_type,imgs in imgs_pths.items():
            (self._imgs_dir / data_type).mkdir(exist_ok=True)
            with open((self._root_dir / (data_type + ".txt")).as_posix(), 'w') as f:
                for img in imgs:
                    f.write((self._imgs_dir / data_type / img.name).as_posix() + '\n')
                    shutil.copy(img, (self._imgs_dir / data_type / img.name).as_posix())           
        
        for data_type,annots in annots_pths.items():
            (self._labels_dir / data_type).mkdir(exist_ok=True)
            for annot in annots:
                shutil.copy(annot, (self._labels_dir / data_type / annot.name))    

    def rectify_pths(self, old_prefix, new_prefix):
        for file in self._root_dir.iterdir():
            lines = []
            if file.as_posix().endswith(".txt"):
                input_file = open(file.as_posix(), "r")
                lines = input_file.readlines()
                new_prefix = old_prefix + "/" + "images" + "/" + file.stem
                lines = [line.replace(old_prefix, new_prefix) for line in lines]
                input_file.close()
                new_file = root_dir.as_posix() + "/" + file.stem + "-version-2" + ".txt"
                output_file = open(new_file, "w")
                output_file.writelines(lines)
                output_file.close()
                    


def flatten_matrix(matrix):
    flat_list = []
    for row in matrix:
        flat_list.extend(row)
    return flat_list


def convert_polygon_to_bbox(polygon):
    polygon_arr = np.array(polygon)
    x_min, x_max = polygon_arr[:, 0].min(), polygon_arr[:, 0].max()
    y_min, y_max = polygon_arr[:, 1].min(), polygon_arr[:, 1].max()
    return (x_max+x_min)/2, (y_max+y_min)/2, x_max-x_min, y_max-y_min #x_min, y_min, x_max, y_max #


if __name__ == "__main__":
    data_pth = Path("/home/mmhamdi/workspace/detection/damages/damages-on-car-parts-detection/car_damages_dataset")
    metadata_pth = data_pth / "meta.json"
    annots_pth = data_pth / "File1" / "ann"
    imgs_pth = data_pth / "File1" / "img"
    root_dir = Path("/home/mmhamdi/workspace/detection/damages/damages-on-car-parts-detection/car_damages_dataset/car_damages_dataset_with_yolov7_format")
    new_imgs_pth =  root_dir / "base"
    labels_pth = root_dir / "labels"
    imgs_dir = root_dir / "images"
    # local_prefix = ... local 
    # cloud_prefix = ... cloud

    data_formatter = DatasetFormatter(data_pth=data_pth, root_dir=root_dir)


    


