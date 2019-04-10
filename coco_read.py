from pycocotools.coco import COCO
from cocostuffapi.PythonAPI.pycocotools.cocostuffhelper import cocoSegmentationToPng
import numpy as np
import imageio as io
from matplotlib import pyplot as plt
from tqdm import tqdm

# coco_things = COCO("../coco/annotations/instances_minival2014.json")
coco_stuff = COCO("stuff/stuff_train2017.json")
image_dir = "../coco/images/train2014/"
image_dir2 = "../coco/images/val2014/"
output_path = "annotations/train/"

# cats = coco_stuff.loadCats(coco_stuff.getCatIds())
# nms = [cat['name'] for cat in cats]
# print('COCO STUFF categories: \n{}\n'.format(' '.join(nms)))
#
# nms = set([cat['supercategory'] for cat in cats])
# print('COCO STUFF supercategories: \n{}'.format(' '.join(nms)))
#
#
# cats = coco_things.loadCats(coco_things.getCatIds())
# nms = [cat['name'] for cat in cats]
# print('COCO THINGS categories: \n{}\n'.format(' '.join(nms)))
#
# nms = set([cat['supercategory'] for cat in cats])
# print('COCO THINGS supercategories: \n{}'.format(' '.join(nms)))

imgIds = coco_stuff.getImgIds()
for img_id in tqdm(imgIds):
	img = coco_stuff.loadImgs(img_id)[0]
	try:
		I = io.imread(image_dir + "COCO_train2014_" + img['file_name'])
	except:
		try:
			I = io.imread(image_dir2 + "COCO_val2014_" + img['file_name'])
		except:
			print("Can't find", img['file_name'])
			continue
	# plt.subplot(1, 3, 1)
	# plt.axis('off')
	# plt.imshow(I)
	# plt.show()

	# plt.subplot(1, 3, 2)
	# plt.imshow(I)
	# plt.axis('off')
	annIds = coco_stuff.getAnnIds(imgIds=img['id'])
	anns = coco_stuff.loadAnns(annIds)
	cocoSegmentationToPng(coco_stuff, img['id'], output_path + img['file_name'][:-4] + "_labelIds" + ".png")
	# coco_stuff.showAnns(anns)

	# plt.subplot(1, 3, 3)
	# plt.imshow(I)
	# plt.axis('off')
	# annIds = coco_things.getAnnIds(imgIds=img['id'])
	# anns = coco_things.loadAnns(annIds)
	# coco_things.showAnns(anns)
