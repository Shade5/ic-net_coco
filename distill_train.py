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
from torch.optim import SGD, lr_scheduler
from tensorboardX import SummaryWriter
from datetime import datetime
from ptsemseg.metrics import runningScore, averageMeter
from ptsemseg.utils import convert_state_dict


def validate(model, valloader, device, writer, epoch):
	model.eval()
	running_metrics = runningScore(valloader.n_classes)

	for i, (images, labels) in enumerate(valloader):

		images = images.to(device)
		_, outputs = model(images)
		pred = outputs.data.max(1)[1].cpu().numpy()

		gt = labels.numpy()
		running_metrics.update(gt, pred)

	score, class_iou = running_metrics.get_scores()

	writer("val/accuracy", score["Mean Acc : \t"], epoch)
	writer("val/iou", score["Mean IoU : \t"], epoch)


def train(cfg):
	# Setup device
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Setup Dataloader
	data_loader = get_loader(cfg["data"]["dataset"])

	t_loader = data_loader(
		"/home/georgejo/cityscapes",
		is_transform=True,
		split="train",
		img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"])
	)

	v_loader = data_loader(
		"/home/georgejo/cityscapes",
		is_transform=True,
		split="val",
		img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"])
	)

	# img, lbl, image_id = t_loader[67]
	# cv2.imshow("img", np.transpose(img.cpu().numpy(), (1, 2, 0)))
	# cv2.imshow("lbl", 255*lbl.cpu().numpy())
	# cv2.waitKey(0)

	n_classes = t_loader.n_classes
	trainloader = data.DataLoader(
		t_loader,
		batch_size=12,
		num_workers=cfg["training"]["n_workers"],
		shuffle=True,
	)
	valloader = data.DataLoader(
		v_loader,
		batch_size=12,
		num_workers=cfg["training"]["n_workers"],
		shuffle=True,
	)
	run_type = "pair"
	writer = SummaryWriter("tbx/" + run_type + str(datetime.now()))

	# Setup Model
	model_dict = {"arch": "ResNetFCN"}
	model = get_model(model_dict, n_classes, version="cityscapes")
	optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
	scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)

	model.train()
	model.to(device)

	n_step = 0

	for epoch in range(125):
		scheduler.step()
		for (images, target, teacher_feat, teacher_score) in tqdm(trainloader):
			images = images.to(device)
			target = target.to(device)
			teacher_feat = teacher_feat.to(device)
			teacher_score = teacher_score.to(device)

			student_feat, student_score = model(images)
			optimizer.zero_grad()
			loss = model.loss(student_feat, student_score, teacher_feat, teacher_score, target, train_type=run_type)
			writer.add_scalar("train/loss", loss.item(), n_step)
			loss.backward()
			optimizer.step()
			n_step += 1

			if n_step % 250 == 0:
				validate(model, valloader, device, writer, epoch)
				model.train()

		if epoch % 100 == 0 and epoch > 0:
			torch.save(model.state_dict(), "checkpoints/" + run_type + str(epoch) + ".pth")


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
