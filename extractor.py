import os
import yaml
import time
import shutil
import torch
import random
import argparse
import numpy as np
import cv2

from torch.utils import data
from tqdm import tqdm

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader
from ptsemseg.utils import get_logger, convert_state_dict
import pickle
from skimage.transform import resize
import imageio as io


def train(cfg):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Dataloader
    data_loader = get_loader(cfg["data"]["dataset"])

    t_loader = data_loader(
        cfg["data"]["path"],
        is_transform=True,
        split="train",
        img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"])
    )

    # img, lbl, image_id = t_loader[67]
    # cv2.imshow("img", np.transpose(img.cpu().numpy(), (1, 2, 0)))
    # cv2.imshow("lbl", 255*lbl.cpu().numpy())
    # cv2.waitKey(0)

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(
        t_loader,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["n_workers"],
        shuffle=False,
    )

    # Setup Model
    model_dict = {"arch": "pspnet"}
    model = get_model(model_dict, n_classes, version="cityscapes")
    state = convert_state_dict(torch.load(cfg["training"]["model_path"])["model_state"])
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    features = {}

    model.eval()
    for (images, labels, image_id) in tqdm(trainloader):
        images = images.to(device)
        outputs = model(images)
        pyramid_pooling = model.px.cpu().detach().numpy()
        cbr_final = model.cx.cpu().detach().numpy()
        classification = model.clx.cpu().detach().numpy()
        interpolated_output = outputs.cpu().detach().numpy()

        for i in range(outputs.shape[0]):
            features[image_id[i]] = {'pyramid_pooling': pyramid_pooling[i], 'cbr_final': cbr_final[i], 'classification': classification[i], 'interpolated_output': interpolated_output[i]}
            # # View output
            # pred = outputs[i].data.max(0)[1].cpu().numpy()
            # pred = pred.astype(np.float32)
            # # float32 with F mode, resize back to orig_size
            # pred = np.round(resize(pred, (t_loader.img_size[0], t_loader.img_size[1])))
            # decoded = t_loader.decode_segmap(pred)
            # io.imwrite("a.png", decoded)

    with open('features.pkl', 'wb') as f:
        pickle.dump(features, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/coco.yml",
        help="Configuration file to use",
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    train(cfg)
