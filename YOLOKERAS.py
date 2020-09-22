# from yolo import YOLO
from PIL import Image
from tensorflow.keras.models import load_model
import cv2,numpy as np
import os
import colorsys
import tensorflow as tf
from PIL import Image, ImageFont, ImageDraw
from tensorflow.keras import backend as K


# img = input('Input image filename:')
# yolo=YOLO()
# img="Data/cat.jpeg"
# image = Image.open(img)
# r_image = yolo.detect_image(image)
# r_image.show()

# define params
img_name="Data/cat.jpeg"
Model_Base_Path="/media/aichunks/Workspace/python/Models/KerasModels/yolomodel/"
model_path=Model_Base_Path+"yolo.h5"
anchors_path=Model_Base_Path +'yolo_anchors.txt'
classes_path= Model_Base_Path +'coco_classes.txt'

model_image_size= (416, 416)

def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

def processImage():
	modlepath=Model_Base_Path+"yoloModel.h5"
	# modlepath="/media/aichunks/Workspace/python/Models/KerasModels/models/full_yolo_backend.h5"
	from tensorflow.keras.models import load_model

	model=load_model(modlepath)
	model.load_weights(Model_Base_Path+"yolo.h5")
	# print(model.summary())
	img=cv2.imread(img_name)
	# img1=cv2.imread("Data/dog.jpeg")
	# img=np.expand_dims(cv2.resize(img,(640,640)),axis=0)
	img=np.expand_dims(cv2.resize(img,(416,416)),axis=0)
	# img1=np.expand_dims(cv2.resize(img1,(640,640)),axis=0)
	# img=np.concatenate([img,img1])
	print(img.shape)
	out=model.predict(img)
	print(len(out),out[0].shape)
	print([o.shape for o in out])
	np.savez("Out2",*out)
	return out

class PostProcessing(tf.keras.layers.Layer):
	def __init__(self,image_shape,anchors_path,classes_path):
		super(PostProcessing, self).__init__()
		self.image_shape=image_shape
		self.max_boxes = 20
		self.score_threshold = .6
		self.iou_threshold = .5
		self.anchors = self._get_anchors(anchors_path)
		self.class_names = self._get_class(classes_path)
		self.num_anchors = len(self.anchors)
		self.num_classes = len(self.class_names)
		self.hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]
		self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), self.hsv_tuples))
		self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
		np.random.seed(10101)  # Fixed seed for consistent colors across runs.
		np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
		np.random.seed(None)  # Reset seed to default.


	def _get_class(self,classes_path):
		classes_path = os.path.expanduser(classes_path)
		with open(classes_path) as f:
			class_names = f.readlines()
		class_names = [c.strip() for c in class_names]
		return class_names


	def _get_anchors(self,anchors_path):
		anchors_path = os.path.expanduser(anchors_path)
		with open(anchors_path) as f:
			anchors = f.readline()
		anchors = [float(x) for x in anchors.split(',')]
		return np.array(anchors).reshape(-1, 2)

	def yolo_head(self,feats, anchors, num_classes, input_shape, calc_loss=False):
		"""Convert final layer features to bounding box parameters."""
		num_anchors = len(anchors)
		# Reshape to batch, height, width, num_anchors, box_params.
		anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

		grid_shape = K.shape(feats)[1:3] # height, width
		grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
			[1, grid_shape[1], 1, 1])
		grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
			[grid_shape[0], 1, 1, 1])
		grid = K.concatenate([grid_x, grid_y])
		grid = K.cast(grid, K.dtype(feats))

		feats = K.reshape(
			feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

		# Adjust preditions to each spatial grid point and anchor size.
		box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
		box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
		box_confidence = K.sigmoid(feats[..., 4:5])
		box_class_probs = K.sigmoid(feats[..., 5:])

		if calc_loss == True:
			return grid, feats, box_xy, box_wh
		return box_xy, box_wh, box_confidence, box_class_probs


	def yolo_correct_boxes(self,box_xy, box_wh, input_shape, image_shape):
		'''Get corrected boxes'''
		box_yx = box_xy[..., ::-1]
		box_hw = box_wh[..., ::-1]
		input_shape = K.cast(input_shape, K.dtype(box_yx))
		image_shape = K.cast(image_shape, K.dtype(box_yx))
		new_shape = K.round(image_shape * K.min(input_shape/image_shape))
		offset = (input_shape-new_shape)/2./input_shape
		scale = input_shape/new_shape
		box_yx = (box_yx - offset) * scale
		box_hw *= scale

		box_mins = box_yx - (box_hw / 2.)
		box_maxes = box_yx + (box_hw / 2.)
		boxes =  K.concatenate([
			box_mins[..., 0:1],  # y_min
			box_mins[..., 1:2],  # x_min
			box_maxes[..., 0:1],  # y_max
			box_maxes[..., 1:2]  # x_max
		])

		# Scale boxes back to original image shape.
		boxes *= K.concatenate([image_shape, image_shape])
		return boxes


	def yolo_boxes_and_scores(self,feats, anchors, num_classes, input_shape, image_shape):
		'''Process Conv layer output'''
		box_xy, box_wh, box_confidence, box_class_probs = self.yolo_head(feats,
			anchors, num_classes, input_shape)
		boxes = self.yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
		boxes = K.reshape(boxes, [-1, 4])
		box_scores = box_confidence * box_class_probs
		box_scores = K.reshape(box_scores, [-1, num_classes])
		return boxes, box_scores

	def call(self,yolo_outputs):
		num_layers = len(yolo_outputs)
		anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]  # default setting
		input_shape = K.shape(yolo_outputs[0])[1:3] * 32
		boxes = []
		box_scores = []
		for l in range(num_layers):
			_boxes, _box_scores = self.yolo_boxes_and_scores(yolo_outputs[l],
				self.anchors[anchor_mask[l]], self.num_classes, input_shape, self.image_shape)
			boxes.append(_boxes)
			box_scores.append(_box_scores)
		boxes = K.concatenate(boxes, axis=0)
		box_scores = K.concatenate(box_scores, axis=0)

		mask = box_scores >= self.score_threshold
		max_boxes_tensor = K.constant(self.max_boxes, dtype='int32')
		boxes_ = []
		scores_ = []
		classes_ = []
		for c in range(self.num_classes):
			# TODO: use keras backend instead of tf.
			class_boxes = tf.boolean_mask(boxes, mask[:, c])
			class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
			nms_index = tf.image.non_max_suppression(
				class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=self.iou_threshold)
			class_boxes = K.gather(class_boxes, nms_index)
			class_box_scores = K.gather(class_box_scores, nms_index)
			classes = K.ones_like(class_box_scores, 'int32') * c
			boxes_.append(class_boxes)
			scores_.append(class_box_scores)
			classes_.append(classes)
		boxes_ = K.concatenate(boxes_, axis=0)
		scores_ = K.concatenate(scores_, axis=0)
		classes_ = K.concatenate(classes_, axis=0)
		return boxes_,scores_,classes_


img=Image.open(img_name)
image_shape=img.size[::-1]
img=letterbox_image(img,model_image_size)
input_shape=img.size[::-1]
image_data = np.array(img, dtype='float32')
image_data /= 255.
image_data = np.expand_dims(image_data, 0)

def defineModel():
	basemodel=load_model(model_path)
	ps=PostProcessing(image_shape,anchors_path,classes_path)
	out_boxes, out_scores, out_classes=ps(basemodel.output)
	model=tf.keras.Model(inputs=[basemodel.input], outputs=[out_boxes, out_scores, out_classes])
	return model

model=defineModel()
out_boxes, out_scores, out_classes=model.predict(image_data)




print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

image = Image.open(img_name)
sz=np.floor(3e-2 * image.size[1] + 0.5).astype('int32')
font = ImageFont.truetype(font="arial.ttf",size=sz)
thickness = (image.size[0] + image.size[1]) // 300

for i, c in reversed(list(enumerate(out_classes))):
    predicted_class = ps.class_names[c]
    box = out_boxes[i]
    score = out_scores[i]

    label = '{} {:.2f}'.format(predicted_class, score)
    draw = ImageDraw.Draw(image)
    label_size = draw.textsize(label, font)

    top, left, bottom, right = box
    top = max(0, np.floor(top + 0.5).astype('int32'))
    left = max(0, np.floor(left + 0.5).astype('int32'))
    bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
    right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
    print(label, (left, top), (right, bottom))

    if top - label_size[1] >= 0:
        text_origin = np.array([left, top - label_size[1]])
    else:
        text_origin = np.array([left, top + 1])

    # My kingdom for a good redistributable image drawing library.
    for i in range(thickness):
        draw.rectangle(
            [left + i, top + i, right - i, bottom - i],
            outline=ps.colors[c])
    draw.rectangle(
        [tuple(text_origin), tuple(text_origin + label_size)],
        fill=ps.colors[c])
    draw.text(text_origin, label, fill=(0, 0, 0), font=font)
    del draw


image.save("DogOut.jpeg")
