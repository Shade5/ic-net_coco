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
from torch.optim import SGD
from tensorboardX import SummaryWriter
from datetime import datetime


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

	writer = SummaryWriter("tbx/regular" + str(datetime.now()))

	# Setup Model
	model_dict = {"arch": "ResNetFCN"}
	model = get_model(model_dict, n_classes, version="cityscapes")
	optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)

	model.train()
	model.to(device)

	n_step = 0

	for epoch in range(10):
		for (images, target, teacher_feat, teacher_score) in tqdm(trainloader):
			images = images.to(device)
			target = target.to(device)
			teacher_feat = teacher_feat.to(device)
			teacher_score = teacher_score.to(device)

			student_feat, student_score = model(images)
			optimizer.zero_grad()
			loss = model.loss(student_feat, student_score, teacher_feat, teacher_score, target, train_type="regular")
			writer.add_scalar("train/loss", loss, n_step)
			loss.backwards()
			optimizer.step()
			n_step += 1

		if epoch % 100 == 0 and epoch > 0:
			torch.save(model.state_dict(), "checkpoints/" + str(epoch) + ".pth")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="config")
	parser.add_argument(
		"--config",
		nargs="?",
		type=str,
		default="configs/resnet_cityscapes.yml",
		help="Configuration file to use",
	)

	args = parser.parse_args()

	with open(args.config) as fp:
		cfg = yaml.load(fp)

	train(cfg)
