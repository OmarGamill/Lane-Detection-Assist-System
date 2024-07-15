import cv2
import math
import torch
import scipy.special
import numpy as np
import torchvision
import torchvision.transforms as transforms
from sklearn.linear_model import LinearRegression
from PIL import Image
from enum import Enum
from scipy.spatial.distance import cdist

from UlterFastLaneDetection.utilities.model import parsingNet

lane_colors = [(0,0,255),(0,255,0),(255,0,0),(0,255,255)]

tusimple_row_anchor = [ 64,  68,  72,  76,  80,  84,  88,  92,  96, 100, 104, 108, 112,
			116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
			168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
			220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
			272, 276, 280, 284]
culane_row_anchor = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]


class ModelType(Enum):
	TUSIMPLE = 0
	CULANE = 1

class ModelConfig():

	def __init__(self, model_type):

		if model_type == ModelType.TUSIMPLE:
			self.init_tusimple_config()
		else:
			self.init_culane_config()

	def init_tusimple_config(self):
		self.img_w = 1280
		self.img_h = 720
		self.row_anchor = tusimple_row_anchor
		self.griding_num = 100
		self.cls_num_per_lane = 56

	def init_culane_config(self):
		self.img_w = 1640
		self.img_h = 590
		self.row_anchor = culane_row_anchor
		self.griding_num = 200
		self.cls_num_per_lane = 18

class UltrafastLaneDetector():

	def __init__(self, model_path, model_type=ModelType.TUSIMPLE, use_gpu=False):

		self.use_gpu = use_gpu

		# Load model configuration based on the model type
		self.cfg = ModelConfig(model_type)

		# Initialize model
		self.model = self.initialize_model(model_path, self.cfg, use_gpu)

		# Initialize image transformation
		self.img_transform = self.initialize_image_transform()

	@staticmethod
	def initialize_model(model_path, cfg, use_gpu):

		# Load the model architecture
		net = parsingNet(pretrained = False, backbone='large', cls_dim = (cfg.griding_num+1,cfg.cls_num_per_lane,4),
						use_aux=False) # we dont need auxiliary segmentation in testing


		# Load the weights from the downloaded model
		if use_gpu:
			if torch.backends.mps.is_built():
				net = net.to("mps")
				state_dict = torch.load(model_path, map_location='mps')['model'] # Apple GPU
			else:
				net = net.cuda()
				state_dict = torch.load(model_path, map_location='cuda')['model'] # CUDA
		else:
			state_dict = torch.load(model_path, map_location='cpu')['model'] # CPU

		compatible_state_dict = {}
		for k, v in state_dict.items():
			if 'module.' in k:
				compatible_state_dict[k[7:]] = v
			else:
				compatible_state_dict[k] = v

		# Load the weights into the model
		net.load_state_dict(compatible_state_dict, strict=False)
		net.eval()

		return net

	@staticmethod
	def initialize_image_transform():
		# Create transfom operation to resize and normalize the input images
		img_transforms = transforms.Compose([
			transforms.Resize((288, 800)),
			transforms.ToTensor(),
			transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
		])

		return img_transforms

	def detect_lanes(self, image, draw_points=True):

		input_tensor = self.prepare_input(image)

		# Perform inference on the image
		output = self.inference(input_tensor)
    # Process output data
		self.lanes_points, self.lanes_detected = self.process_output(output, self.cfg)


		return self.lanes_points, self.lanes_detected

	def prepare_input(self, img):
		# Transform the image for inference
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img_pil = Image.fromarray(img)
		input_img = self.img_transform(img_pil)
		input_tensor = input_img[None, ...]

		if self.use_gpu:
			if not torch.backends.mps.is_built():
				input_tensor = input_tensor.cuda()

		return input_tensor

	def inference(self, input_tensor):
		with torch.no_grad():
			output = self.model(input_tensor)

		return output

	@staticmethod
	def process_output(output, cfg):		
		# Parse the output of the model
		processed_output = output[0].data.cpu().numpy()
		processed_output = processed_output[:, ::-1, :]
		prob = scipy.special.softmax(processed_output[:-1, :, :], axis=0)
		idx = np.arange(cfg.griding_num) + 1
		idx = idx.reshape(-1, 1, 1)
		loc = np.sum(prob * idx, axis=0)
		processed_output = np.argmax(processed_output, axis=0)
		loc[processed_output == cfg.griding_num] = 0
		processed_output = loc


		col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
		col_sample_w = col_sample[1] - col_sample[0]

		lanes_points = []
		lanes_detected = []

		max_lanes = processed_output.shape[1]
		for lane_num in range(max_lanes):
			lane_points = []
			# Check if there are any points detected in the lane
			if np.sum(processed_output[:, lane_num] != 0) > 2:

				lanes_detected.append(True)

				# Process each of the points for each lane
				for point_num in range(processed_output.shape[0]):
					if processed_output[point_num, lane_num] > 0:
						lane_point = [int(processed_output[point_num, lane_num] * col_sample_w * cfg.img_w / 800) - 1, int(cfg.img_h * (cfg.row_anchor[cfg.cls_num_per_lane-1-point_num]/288)) - 1 ]
						lane_points.append(lane_point)
			else:
				lanes_detected.append(False)

			lanes_points.append(lane_points)
		return lanes_points, np.array(lanes_detected)

	@staticmethod
	def draw_lanes(input_img, lanes_points, lanes_detected, cfg_w, cfg_h, draw_points=True):
		# Write the detected line points in the image
		visualization_img = cv2.resize(input_img, (cfg_w, cfg_h), interpolation = cv2.INTER_AREA)
		'''
		# Draw a mask for the current lane
		if(lanes_detected[1] and lanes_detected[2]):
			lane_segment_img = visualization_img.copy()
			
			cv2.fillPoly(lane_segment_img, pts = [np.vstack((lanes_points[1],np.flipud(lanes_points[2])))], color =(255,191,0))
			visualization_img = cv2.addWeighted(visualization_img, 0.7, lane_segment_img, 0.3, 0)

		
		'''
		
		if(draw_points):
			for lane_num,lane_points in enumerate(lanes_points):
				for lane_point in lane_points:
					cv2.circle(visualization_img, (lane_point[0],lane_point[1]), 3, lane_colors[lane_num], -1)

		return visualization_img
	
	@staticmethod
	def prosess_points(lane_dete, points):
		max_points, idx, x , y = 0,-1,[], []
		for i in range(4):
			if lane_dete[i]:
				if max_points < len(points[i]):
					max_points=len(points[i])
					idx = i
		
		if idx != -1:
			for x_ , y_ in points[idx]:
				
				x.append(x_)
				y.append(y_)
		
		#print('idx', idx)
		return np.array(x).reshape(-1,1),np.array(y).reshape(-1,1)

			

	@staticmethod
	def steering_angle(x, y ):
		model = LinearRegression(fit_intercept=True)
		
		model.fit(x,y)
		
		#print(model.intercept_)
		angle_red = math.degrees(np.arctan(model.coef_)/2 ) #this line after update code 
		m = model.coef_[0][0]
		angle_rad = np.arctan(m)
		angle_deg = np.degrees(angle_rad)
		 
		
		return angle_deg 
	
	@staticmethod
	def calculate_midpoints(left_lane_points, right_lane_points):
		midpoints = []
		print(left_lane_points, '\njjj',right_lane_points)
		for left_point, right_point in zip(left_lane_points, right_lane_points):
			midpoint = ((left_point[0] + right_point[0]) / 2, (left_point[1] + right_point[1]) / 2)
			midpoints.append(midpoint)
		return midpoints
	@staticmethod
	def calculate_direction_angle(midpoints):
		# Use the first and last midpoint to calculate the direction
		first_midpoint = midpoints[0]
		last_midpoint = midpoints[-1]
		
		# Calculate the difference in x and y
		delta_x = last_midpoint[0] - first_midpoint[0]
		delta_y = last_midpoint[1] - first_midpoint[1]
		
		# Calculate the angle in radians
		angle_radians = math.atan2(delta_y, delta_x)
		
		# Convert the angle to degrees
		angle_degrees = math.degrees(angle_radians)
		
		return angle_degrees

	@staticmethod
	def angle(lane_dete, points):
		left_lane_points, right_lane_points = UltrafastLaneDetector.prosess_points(lane_dete, points)
		midpoints = UltrafastLaneDetector.calculate_midpoints(left_lane_points, right_lane_points)
		angle = 90 - UltrafastLaneDetector.calculate_direction_angle(midpoints)
		return angle
	
	def fit_line_and_angle(x, y):
		"""
		Fits a line y = mx + b to the data points given by x and y and calculates the angle of the line.
		
		Args:
		- x: numpy array of shape (44, 1)
		- y: numpy array of shape (44, 1)
		
		Returns:
		- m: Slope of the fitted line
		- b: Intercept of the fitted line
		- angle: Angle of the fitted line in degrees with respect to the x-axis
		"""
		# Ensure x and y are 1D arrays
		x = x.reshape(-1)
		y = y.reshape(-1)
		
		# Design matrix
		A = np.vstack([x, np.ones_like(x)]).T
		
		# Solve for the least squares solution
		m, b = np.linalg.lstsq(A, y, rcond=None)[0]
		
		# Calculate the angle in radians and convert to degrees
		angle_rad = np.arctan(m)
		angle_deg = np.degrees(angle_rad)
		
		return angle_deg

	
        


	







