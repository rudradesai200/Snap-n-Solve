import cv2
import operator
import numpy as np
from keras.models import load_model
from django.conf import settings
import os
import sys

# BASE_DIR = "/home/rudra/Desktop/cfviewer/codeforces_problems_list/"
# model_path = "/Users/rudrapd/repos/personal/Sudoku-Solver/mldeployed/sudoku/mlmodels/my_model.h5"
model_path = os.path.join(settings.BASE_DIR,"sudoku/mlmodels/my_model.h5")
model = load_model(model_path)

def show_digits(digits, colour=255):
	"""Shows list of 81 extracted digits in a grid format"""
	rows = []
	with_border = [cv2.copyMakeBorder(img.copy(), 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, colour) for img in digits]
	for i in range(9):
		row = np.concatenate(with_border[i * 9:((i + 1) * 9)], axis=1)
		rows.append(row)
	return np.concatenate(rows)


def convert_when_colour(colour, img):
	"""Dynamically converts an image to colour if the input colour is a tuple and the image is grayscale."""
	if len(colour) == 3:
		if len(img.shape) == 2:
			img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
		elif img.shape[2] == 1:
			img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	return img

def pre_process_image(img, skip_dilate=False):
	"""Uses a blurring function, adaptive thresholding and dilation to expose the main features of an image."""
	proc = cv2.GaussianBlur(img.copy(), (9, 9), 0)
	proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
	proc = cv2.bitwise_not(proc, proc)
	return proc


def find_corners_of_largest_polygon(img):
	"""Finds the 4 extreme corners of the largest contour in the image.
		Basically the main sudoku grid.
	"""
	contours,_ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
	contours = sorted(contours, key=cv2.contourArea, reverse=True)  # Sort by area, descending
	polygon = contours[0]  # Largest image
	bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
	top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
	bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
	top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
	return [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]


def distance_between(p1, p2):
	"""Returns the scalar distance between two points"""
	a = p2[0] - p1[0]
	b = p2[1] - p1[1]
	return np.sqrt((a ** 2) + (b ** 2))


def crop_and_warp(img, crop_rect):
	"""Crops and warps a rectangular section from an image into a square of similar size."""
	top_left, top_right, bottom_right, bottom_left = crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]
	src = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
	side = max([
		distance_between(bottom_right, top_right),
		distance_between(top_left, bottom_left),
		distance_between(bottom_right, bottom_left),
		distance_between(top_left, top_right)
	])
	dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype=np.float32)
	m = cv2.getPerspectiveTransform(src, dst)
	return cv2.warpPerspective(img, m, (int(side), int(side)))



def infer_grid(img):
	"""Infers 81 cell grid from a square image."""
	squares = []
	side = img.shape[:1]
	side = side[0] / 9
	for j in range(9):
		for i in range(9):
			p1 = (i * side, j * side)  # Top left corner of a bounding box
			p2 = ((i + 1) * side, (j + 1) * side)  # Bottom right corner of bounding box
			squares.append((p1, p2))
	return squares


def cut_from_rect(img, rect):
	"""Cuts a rectangle from an image using the top left and bottom right points."""
	return img[int(rect[0][1]):int(rect[1][1]), int(rect[0][0]):int(rect[1][0])]


def scale_and_centre(img, size, margin=0, background=0):
	"""Scales and centres an image onto a new background square."""
	h, w = img.shape[:2]

	def centre_pad(length):
		"""Handles centering for a given length that may be odd or even."""
		if length % 2 == 0:
			side1 = int((size - length) / 2)
			side2 = side1
		else:
			side1 = int((size - length) / 2)
			side2 = side1 + 1
		return side1, side2

	def scale(r, x):
		return int(r * x)

	if h > w:
		t_pad = int(margin / 2)
		b_pad = t_pad
		ratio = (size - margin) / h
		w, h = scale(ratio, w), scale(ratio, h)
		l_pad, r_pad = centre_pad(w)
	else:
		l_pad = int(margin / 2)
		r_pad = l_pad
		ratio = (size - margin) / w
		w, h = scale(ratio, w), scale(ratio, h)
		t_pad, b_pad = centre_pad(h)

	img = cv2.resize(img, (w, h))
	img = cv2.copyMakeBorder(img, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, None, background)
	return cv2.resize(img, (size, size))


def find_largest_feature(inp_img, scan_tl=None, scan_br=None):
	"""
	Uses the fact the `floodFill` function returns a bounding box of the area it filled to find the biggest
	connected pixel structure in the image. Fills this structure in white, reducing the rest to black.
	"""
	img = inp_img.copy()  # Copy the image, leaving the original untouched
	height, width = img.shape[:2]

	max_area = 0
	seed_point = (None, None)

	if scan_tl is None:
		scan_tl = [0, 0]

	if scan_br is None:
		scan_br = [width, height]

	# Loop through the image
	for x in range(scan_tl[0], scan_br[0]):
		for y in range(scan_tl[1], scan_br[1]):
			# Only operate on light or white squares
			if img.item(y, x) == 255 and x < width and y < height:  # Note that .item() appears to take input as y, x
				area = cv2.floodFill(img, None, (x, y), 64)
				if area[0] > max_area:  # Gets the maximum bound area which should be the grid
					max_area = area[0]
					seed_point = (x, y)

	# Colour everything grey (compensates for features outside of our middle scanning range
	for x in range(width):
		for y in range(height):
			if img.item(y, x) == 255 and x < width and y < height:
				cv2.floodFill(img, None, (x, y), 64)

	mask = np.zeros((height + 2, width + 2), np.uint8)  # Mask that is 2 pixels bigger than the image

	# Highlight the main feature
	if all([p is not None for p in seed_point]):
		cv2.floodFill(img, mask, seed_point, 255)

	top, bottom, left, right = height, 0, width, 0

	for x in range(width):
		for y in range(height):
			if img.item(y, x) == 64:  # Hide anything that isn't the main feature
				cv2.floodFill(img, mask, (x, y), 0)

			# Find the bounding parameters
			if img.item(y, x) == 255:
				top = y if y < top else top
				bottom = y if y > bottom else bottom
				left = x if x < left else left
				right = x if x > right else right

	bbox = [[left, top], [right, bottom]]
	return img, np.array(bbox, dtype='float32'), seed_point


def extract_digit(img, rect, size):
	"""Extracts a digit (if one exists) from a Sudoku square."""

	digit = cut_from_rect(img, rect)  # Get the digit box from the whole square

	# Use fill feature finding to get the largest feature in middle of the box
	# Margin used to define an area in the middle we would expect to find a pixel belonging to the digit
	h, w = digit.shape[:2]
	margin = int(np.mean([h, w]) / 2.5)
	_, bbox, _ = find_largest_feature(digit, [margin, margin], [w - margin, h - margin])
	digit = cut_from_rect(digit, bbox)

	# Scale and pad the digit so that it fits a square of the digit size we're using for machine learning
	w = bbox[1][0] - bbox[0][0]
	h = bbox[1][1] - bbox[0][1]

	# Ignore any small bounding boxes
	if w > 0 and h > 0 and (w * h) > 100 and len(digit) > 0:
		return scale_and_centre(digit, size, 4)
	else:
		return np.zeros((size, size), np.uint8)


def get_digits(img, squares, size):
	"""Extracts digits from their cells and builds an array"""
	digits = []
	img = pre_process_image(img.copy())
	for square in squares:
		digits.append(extract_digit(img, square, size))
	return digits

def createborder(path,color):
    borderType = cv2.BORDER_CONSTANT
    imageName = path
    # Loads an image
    src = cv2.imread(cv2.samples.findFile(imageName), cv2.IMREAD_COLOR)
    # Check if image is loaded fine
    if src is None:
        return False

    top = int(0.008 * src.shape[0])  # shape[0] = rows
    bottom = top
    left = int(0.008 * src.shape[1])  # shape[1] = cols
    right = left
    value = color
    dst = cv2.copyMakeBorder(src, top, bottom, left, right, borderType, None, value)
    cv2.imwrite(path,dst)
    return True

def is_valid_sudoku(board):
	for i in range(9):
		row = {}
		column = {}
		block = {}
		row_cube = 3 * (i//3)
		column_cube = 3 * (i%3)
		for j in range(9):
			if board[i][j]!=0 and board[i][j] in row:
				return False
			row[board[i][j]] = 1
			if board[j][i]!=0 and board[j][i] in column:
				return False
			column[board[j][i]] = 1
			rc= row_cube+j//3
			cc = column_cube + j%3
			if board[rc][cc] in block and board[rc][cc]!=0:
				return False
			block[board[rc][cc]]=1
	return True

def parse_grid(request,path, save_path):
	# Process image to get the cropped sudoku grid
	original = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
	processed = pre_process_image(original)
	corners = find_corners_of_largest_polygon(processed)
	cropped = crop_and_warp(original, corners)

	# Grab the 81 squares out of the sudoku grid
	squares = infer_grid(cropped)

		# Extract digits
	digits = get_digits(cropped, squares, 28)
	digs = show_digits(digits)
	imgs = np.array(digits,dtype=np.float64)
	imgs = np.reshape(imgs,(81,-1))
	x = model.predict(imgs,verbose=0)
	y = (np.argmax(x,axis=1))
	y = np.reshape(y,(9,9))
	if not is_valid_sudoku(y):
		raise ValueError()
	cv2.imwrite(save_path,digs)
	return y,digs