import cv2
import numpy as np
import argparse
import time
import math
import os
import tempfile

# Example usage: python ./src/sort_tracker.py -v input_video.mp4 -s output_video.mp4 

# PARAMÈTRES
# Units: pixels / pixels^2 (area)
MIN_AREA = 500                 # aire minimale (px^2) d'un contour pour qu'il soit considéré en détection
THRESH_VAL = 100               # seuil binaire utilisé pour la détection (0-255)
NMS_IOU = 0.3                  # IoU seuil pour la NMS (fusion de boîtes)
MAX_DISAPPEARED = 1            # frames avant suppression d'un objet absent
MAX_DISTANCE = 200             # distance max (px) pour associer une détection à un objet suivi; <=0 désactive la contrainte (autorise grands sauts)
WATERSHED = True               # activer la séparation des amas via watershed (True/False)
LARGE_AREA_FACTOR = 1.0        # facteur multiplié par MIN_AREA pour considérer un contour "grand" (pour watershed)
WATERSHED_BOTTOM_MARGIN = 20   # éviter d'appliquer watershed si le blob touche le bas (px)
MIN_TRACK_AREA = 1000          # aire minimale (px^2) requise pour qu'une détection devienne un track
MIN_TRACK_SIDE = 0             # côté min (px) pour un track; 0 -> auto = sqrt(MIN_TRACK_AREA)
NEW_IOU_GATE = 0.3             # IoU pour éviter création d'un nouvel ID lorsqu'il recouvre fortement un existant
MIN_TRACK_LENGTH = 5           # nombre minimal de frames pour valider une trajectoire (filtre anti-bruit)

def nms(boxes, scores=None, iou_thresh=0.1):
	"""
	Simple Non-Maximum Suppression for axis-aligned boxes.
	boxes: list of (x1,y1,x2,y2)
	scores: optional list of scores (higher = keep first). If None, uses area.
	"""
	if not boxes:
		return []
	b = np.array(boxes, dtype=np.float32)
	x1 = b[:,0]; y1 = b[:,1]; x2 = b[:,2]; y2 = b[:,3]
	areas = (x2 - x1 + 1) * (y2 - y1 + 1)
	if scores is None:
		order = areas.argsort()[::-1]  # larger first
	else:
		order = np.array(scores).argsort()[::-1]

	keep = []
	while order.size > 0:
		i = order[0]
		keep.append(int(i))
		xx1 = np.maximum(x1[i], x1[order[1:]])
		yy1 = np.maximum(y1[i], y1[order[1:]])
		xx2 = np.minimum(x2[i], x2[order[1:]])
		yy2 = np.minimum(y2[i], y2[order[1:]])

		w = np.maximum(0.0, xx2 - xx1 + 1)
		h = np.maximum(0.0, yy2 - yy1 + 1)
		inter = w * h
		iou = inter / (areas[i] + areas[order[1:]] - inter)
		inds = np.where(iou <= iou_thresh)[0]
		order = order[inds + 1]
	return keep

def bbox_area(b):
	# x1,y1,x2,y2 -> area in pixels^2
	return max(0, int(b[2]) - int(b[0])) * max(0, int(b[3]) - int(b[1]))

def bbox_iou(a, b):
	# axis-aligned IoU
	ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
	ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
	iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
	inter = iw * ih
	union = bbox_area(a) + bbox_area(b) - inter
	return 0.0 if union == 0 else float(inter) / float(union)

def split_mask_watershed(bw, bbox, min_area=100, thresh_rel=0.5, frame_h=None, bottom_margin=20):
	"""
	Split a binary mask region (roi in bw) using distance transform + watershed.
	bw: binary image (0/255) of whole frame
	bbox: (x,y,x2,y2) region to process
	return: list of bbox tuples (global coords) for found segments (filtered by min_area)
	"""
	x1, y1, x2, y2 = bbox
	# avoid splitting if ROI reaches near the bottom edge (merge behavior there is unreliable)
	if (frame_h is not None) and (y2 >= frame_h - bottom_margin):
		return []

	roi = bw[y1:y2, x1:x2]
	if roi.size == 0:
		return []

	# ensure binary 0/255
	_, roi_bin = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY)

	# distance transform
	dist = cv2.distanceTransform(roi_bin, cv2.DIST_L2, 5)
	if dist.max() == 0:
		return []

	# markers: threshold on distance to get sure foreground
	_, sure_fg = cv2.threshold(dist, thresh_rel * dist.max(), 255, 0)
	sure_fg = np.uint8(sure_fg)

	# sure background via dilation
	k = max(3, int(round(math.sqrt(min_area))))
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
	sure_bg = cv2.dilate(roi_bin, kernel, iterations=1)

	# unknown region
	unknown = cv2.subtract(sure_bg, sure_fg)

	# connected components for markers
	num_markers, markers = cv2.connectedComponents(sure_fg)
	if num_markers <= 1:
		# nothing to split
		return []

	# prepare 3-channel image for watershed (use ROI as "image")
	roi_color = cv2.cvtColor(roi_bin, cv2.COLOR_GRAY2BGR)
	# markers need to be int32 and start from 1
	markers = markers + 1
	markers[unknown == 255] = 0

	# apply watershed (will label borders with -1)
	cv2.watershed(roi_color, markers)

	# collect boxes per label (>1)
	boxes = []
	for lab in range(2, markers.max() + 1):
		mask = (markers == lab).astype('uint8') * 255
		conts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		for c in conts:
			a = cv2.contourArea(c)
			if a < min_area:
				continue
			xx, yy, ww, hh = cv2.boundingRect(c)
			boxes.append((int(x1 + xx), int(y1 + yy), int(x1 + xx + ww), int(y1 + yy + hh)))
	return boxes

def detect_rectangles(frame, min_area=100, thresh_val=100, iou_thresh=0.3,
                      use_watershed=False, large_area_factor=4, watershed_bottom_margin=20):
	"""
	Detect white rectangles on black background.
	- Morphological opening/closing to reduce speckle.
	- Filter small contours by area and by minimum side length (min_side).
	- Keep contours that look rectangular (approx or solidity).
	- Remove overlapping boxes with NMS (iou_thresh).
	Returns list of bboxes as (x1, y1, x2, y2).
	"""
	# frame can be color or gray
	if len(frame.shape) == 3:
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	else:
		gray = frame.copy()

	# add: image height used to avoid watershed near bottom
	frame_h = frame.shape[0]

	# binary + morph to reduce noise
	_, bw = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
	k = max(3, int(round(math.sqrt(min_area))))  # kernel size related to min_area
	if k % 2 == 0:
		k += 1
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
	bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
	bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)

	contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	candidates = []
	scores = []

	# minimum side length to reject thin/small boxes (reduces bruit)
	min_side = max(2, int(round(math.sqrt(min_area))))  # you can increase this value if needed

	for cnt in contours:
		area = cv2.contourArea(cnt)
		if area < min_area:
			continue
		x, y, w, h = cv2.boundingRect(cnt)

		# if contour is very large and watershed requested, try to split it (skip if near bottom)
		if use_watershed and area > max(min_area * large_area_factor, min_area * 2):
			if y + h < frame_h - watershed_bottom_margin:
				splits = split_mask_watershed(bw, (x, y, x + w, y + h), min_area=min_area, frame_h=frame_h, bottom_margin=watershed_bottom_margin)
				if splits:
					for sb in splits:
						candidates.append(sb)
						# score by area
						sx1, sy1, sx2, sy2 = sb
						scores.append((sx2 - sx1) * (sy2 - sy1))
					continue  # skip adding original large box

		# reject boxes that are too small in either dimension
		if w < min_side or h < min_side:
			continue

		# approximate polygon to check rectangularity
		peri = cv2.arcLength(cnt, True)
		approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
		solidity = area / float(w * h) if w * h > 0 else 0
		keep = False
		if len(approx) == 4:
			keep = True
		elif solidity > 0.5:
			keep = True
		if not keep:
			continue
		box = (int(x), int(y), int(x + w), int(y + h))
		candidates.append(box)
		scores.append(area)  # larger contours preferred

	# remove overlapping boxes
	keep_idx = nms(candidates, scores=scores, iou_thresh=iou_thresh)
	bboxes = [candidates[i] for i in keep_idx]
	return bboxes

class SortTracker:
	"""
	Simple centroid-based tracker that assigns persistent IDs to bboxes.
	Usage:
		tracker = SortTracker(max_disappeared=5, max_distance=50)
		detections = detect_rectangles(frame)
		objects = tracker.update(detections)  # returns dict id -> bbox
		frame = tracker.annotate(frame, objects)
	"""
	def __init__(self, max_disappeared=5, max_distance=None, min_track_area=100, min_track_side=None, new_iou_gate=0.6):
		"""
		max_distance: if None -> no distance gating (allow large jumps).
		If a non-positive value is provided (<=0), it is treated as None.
		"""
		self.next_object_id = 0
		self.objects = {}       # id -> bbox (x1,y1,x2,y2)
		# store centroids as floats to allow sub-pixel prediction
		self.centroids = {}     # id -> (cx,cy) as floats
		self.velocities = {}    # id -> (vx,vy) predicted per-frame displacement
		self.disappeared = {}   # id -> frames disappeared
		self.max_disappeared = max_disappeared
		# normalize max_distance: <=0 -> None
		if max_distance is not None and isinstance(max_distance, (int, float)) and max_distance <= 0:
			max_distance = None
		self.max_distance = max_distance
		self.trajectories = {}     # id -> list of (frame_idx, bbox, centroid)
		self.completed = {}        # finished trajectories (id -> list)
		# new params
		self.min_track_area = min_track_area
		if min_track_side is None or min_track_side <= 0:
			self.min_track_side = max(2, int(round(math.sqrt(min_track_area))))
		else:
			self.min_track_side = min_track_side
		self.new_iou_gate = new_iou_gate

	def register(self, bbox, frame_idx=None):
		# skip registering too-small detections
		if bbox_area(bbox) < self.min_track_area:
			return
		x1, y1, x2, y2 = bbox
		w = x2 - x1; h = y2 - y1
		if w < self.min_track_side or h < self.min_track_side:
			return
		# skip if overlaps strongly an existing object (avoid duplicate IDs)
		for oid, ob in self.objects.items():
			if bbox_iou(bbox, ob) > self.new_iou_gate:
				return
		# register new object
		self.objects[self.next_object_id] = bbox
		cx = (x1 + x2) / 2.0; cy = (y1 + y2) / 2.0
		self.centroids[self.next_object_id] = (cx, cy)
		self.velocities[self.next_object_id] = (0.0, 0.0)
		self.disappeared[self.next_object_id] = 0
		self.trajectories[self.next_object_id] = []
		# append initial observation if frame_idx provided
		if frame_idx is not None:
			cx = (bbox[0] + bbox[2]) / 2.0
			cy = (bbox[1] + bbox[3]) / 2.0
			self.trajectories[self.next_object_id].append((frame_idx, bbox, (cx, cy)))
		self.next_object_id += 1

	def deregister(self, object_id):
		# before deletion, store final trajectory
		if object_id in self.trajectories:
			self.completed[object_id] = self.trajectories.pop(object_id)
		# remove velocity too
		if object_id in self.objects:
			del self.objects[object_id]
		if object_id in self.centroids:
			del self.centroids[object_id]
		if object_id in self.velocities:
			del self.velocities[object_id]
		if object_id in self.disappeared:
			del self.disappeared[object_id]

	def update(self, detections, frame_idx=None):
		"""
		detections: list of (x1,y1,x2,y2)
		frame_idx: optional integer index of current frame (used to store trajectories)
		returns dict of tracked objects: id -> bbox
		"""
		if len(detections) == 0:
			# mark all existing as disappeared, but also predict their centroid forward
			for oid in list(self.disappeared.keys()):
				# predict centroid forward by velocity so the "center moves" while missing
				if oid in self.centroids and oid in self.velocities:
					cx, cy = self.centroids[oid]
					vx, vy = self.velocities[oid]
					self.centroids[oid] = (cx + vx, cy + vy)
				self.disappeared[oid] += 1
				if self.disappeared[oid] > self.max_disappeared:
					self.deregister(oid)
			return dict(self.objects)

		# compute centroids for detections (as floats)
		input_centroids = []
		for (x1, y1, x2, y2) in detections:
			input_centroids.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))

		# if no existing objects, register all detections
		if len(self.centroids) == 0:
			for bbox in detections:
				if bbox_area(bbox) >= self.min_track_area:
					x1, y1, x2, y2 = bbox
					w = x2 - x1; h = y2 - y1
					if w >= self.min_track_side and h >= self.min_track_side:
						self.register(bbox, frame_idx=frame_idx)
			return dict(self.objects)

		# build cost matrix (Euclidean distances) using predicted centroids
		object_ids = list(self.centroids.keys())
		object_centroids = []
		for oid in object_ids:
			cx, cy = self.centroids[oid]
			vx, vy = self.velocities.get(oid, (0.0, 0.0))
			# predicted position (allows center to move between frames)
			pred_x = cx + vx
			pred_y = cy + vy
			object_centroids.append((pred_x, pred_y))

		D = np.zeros((len(object_centroids), len(input_centroids)), dtype=np.float32)
		for i, oc in enumerate(object_centroids):
			for j, ic in enumerate(input_centroids):
				D[i, j] = np.hypot(oc[0] - ic[0], oc[1] - ic[1])

		# try Hungarian assignment, else greedy fallback
		rows = cols = []
		try:
			from scipy.optimize import linear_sum_assignment
			rows, cols = linear_sum_assignment(D)
		except Exception:
			# greedy: repeatedly pick smallest distance
			rows, cols = [], []
			D_copy = D.copy()
			used_rows = set()
			used_cols = set()
			while True:
				min_idx = np.unravel_index(np.argmin(D_copy), D_copy.shape)
				min_val = D_copy[min_idx]
				if np.isinf(min_val):
					break
				r, c = min_idx
				if r in used_rows or c in used_cols:
					D_copy[r, c] = np.inf
					continue
				rows.append(r); cols.append(c)
				used_rows.add(r); used_cols.add(c)
				D_copy[r, :] = np.inf
				D_copy[:, c] = np.inf
				if len(used_rows) == D.shape[0] or len(used_cols) == D.shape[1]:
					break

		# keep track of matched and unmatched
		assigned_rows = set()
		assigned_cols = set()
		for r, c in zip(rows, cols):
			# only apply gating if max_distance is set
			if (self.max_distance is not None) and (D[r, c] > self.max_distance):
				# treat as unassigned (too far)
				continue
			object_id = object_ids[r]
			# compute new velocity (EMA) based on difference between measured centroid and previous centroid
			prev_cx, prev_cy = self.centroids[object_id]
			new_cx, new_cy = input_centroids[c]
			raw_vx = new_cx - prev_cx
			raw_vy = new_cy - prev_cy
			old_vx, old_vy = self.velocities.get(object_id, (0.0, 0.0))
			# smoothing factor (tunable)
			alpha = 0.6
			vx = alpha * raw_vx + (1.0 - alpha) * old_vx
			vy = alpha * raw_vy + (1.0 - alpha) * old_vy
			self.velocities[object_id] = (vx, vy)

			# update stored centroid to measured centroid (float)
			self.centroids[object_id] = (new_cx, new_cy)

			# update bbox and reset disappeared counter
			self.objects[object_id] = detections[c]
			self.disappeared[object_id] = 0
			assigned_rows.add(r)
			assigned_cols.add(c)

			# append to trajectory
			if frame_idx is not None:
				self.trajectories.setdefault(object_id, []).append((frame_idx, detections[c], (new_cx, new_cy)))

		# process unassigned existing objects: predict centroid forward and increase disappeared
		for r in range(D.shape[0]):
			if r not in assigned_rows:
				object_id = object_ids[r]
				# predict forward by velocity (so center moves)
				if object_id in self.centroids:
					cx, cy = self.centroids[object_id]
					vx, vy = self.velocities.get(object_id, (0.0, 0.0))
					self.centroids[object_id] = (cx + vx, cy + vy)
				self.disappeared[object_id] += 1
				if self.disappeared[object_id] > self.max_disappeared:
					self.deregister(object_id)

		# register new detections (unassigned cols) but only if large enough and not overlapping existing objects
		for c in range(D.shape[1]):
			if c not in assigned_cols:
				bbox = detections[c]
				if bbox_area(bbox) < self.min_track_area:
					continue
				x1, y1, x2, y2 = bbox
				w = x2 - x1; h = y2 - y1
				if w < self.min_track_side or h < self.min_track_side:
					continue
				overlap = any(bbox_iou(bbox, ob) > self.new_iou_gate for ob in self.objects.values())
				if overlap:
					continue
				self.register(bbox, frame_idx=frame_idx)

		return dict(self.objects)

	def finalize(self):
		"""
		Call at end of processing to move active trajectories to completed.
		"""
		for oid in list(self.trajectories.keys()):
			self.completed[oid] = self.trajectories.pop(oid)
		# clear running maps
		self.objects.clear()
		self.centroids.clear()
		self.velocities.clear()
		self.disappeared.clear()

	def filter_trajectories(self, min_length=3):
		"""
		Keep only trajectories with length >= min_length (number of frames).
		Returns set of valid ids.
		"""
		valid = set()
		for oid, traj in self.completed.items():
			if len(traj) >= min_length:
				valid.add(oid)
		# also consider still-active ones (if any)
		for oid, traj in self.trajectories.items():
			if len(traj) >= min_length:
				valid.add(oid)
		return valid

	def annotate(self, frame, objects=None, box_color=(0,255,0), text_color=(0,255,0)):
		"""
		Draw bounding boxes and IDs on frame. objects can be provided or will use internal state.
		"""
		if objects is None:
			objects = self.objects
		out = frame.copy()
		for oid, bbox in objects.items():
			x1, y1, x2, y2 = bbox
			cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 2)
			cv2.putText(out, str(oid), (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
		return out

def parse_args():
	p = argparse.ArgumentParser(description="Track white rectangles on black background (integrated main)")
	p.add_argument("--video", "-v", default="0", help="Path to video file or '0' for webcam")
	p.add_argument("--min_area", type=int, default=MIN_AREA, help=f"Minimum contour area to keep (px^2). default={MIN_AREA}")
	p.add_argument("--thresh", type=int, default=THRESH_VAL, help=f"Binary threshold value. default={THRESH_VAL}")
	p.add_argument("--iou", type=float, default=NMS_IOU, help=f"IoU threshold for NMS. default={NMS_IOU}")
	p.add_argument("--save", "-s", help="Path to save annotated output (optional)")
	p.add_argument("--max_disappeared", type=int, default=MAX_DISAPPEARED, help=f"Frames before a missing object is deregistered. default={MAX_DISAPPEARED}")
	p.add_argument("--max_distance", type=float, default=MAX_DISTANCE,
	               help=f"Max matching distance in pixels. <=0 disables distance gating (allows large jumps). default={MAX_DISTANCE}")
	p.add_argument("--watershed", action="store_true" if WATERSHED else "store_false", help=f"Try to split large merged blobs with watershed. default={WATERSHED}")
	p.add_argument("--large_area_factor", type=float, default=LARGE_AREA_FACTOR, help=f"Factor above min_area to consider a contour 'large' for watershed. default={LARGE_AREA_FACTOR}")
	p.add_argument("--min_track_length", type=int, default=MIN_TRACK_LENGTH, help=f"Minimum trajectory length (frames) to be considered valid. default={MIN_TRACK_LENGTH}")
	p.add_argument("--save_validated", help="Path to save annotated video with only validated tracks (optional)")
	p.add_argument("--min_track_area", type=int, default=MIN_TRACK_AREA, help=f"Minimum area (px^2) for a detection to become a tracked object. default={MIN_TRACK_AREA}")
	p.add_argument("--min_track_side", type=int, default=MIN_TRACK_SIDE, help=f"Minimum side length (px) for tracked boxes. 0 = auto from area. default={MIN_TRACK_SIDE}")
	p.add_argument("--watershed_bottom_margin", type=int, default=WATERSHED_BOTTOM_MARGIN, help=f"Pixels from bottom where watershed splitting is disabled. default={WATERSHED_BOTTOM_MARGIN}")
	p.add_argument("--new_iou_gate", type=float, default=NEW_IOU_GATE, help=f"IoU threshold to avoid creating duplicate new IDs. default={NEW_IOU_GATE}")
	# new: path to the real color video to use for extracting crops (optional)
	p.add_argument("--color", "-c", default=None, help="Path to real color video to extract crops from (optional). If omitted uses --video")
	return p.parse_args()

def open_capture(src):
	try:
		idx = int(src)
		return cv2.VideoCapture(idx)
	except Exception:
		return cv2.VideoCapture(src)

def main():
	args = parse_args()
	cap = open_capture(args.video)
	if not cap.isOpened():
		print("Erreur: impossible d'ouvrir la source", args.video)
		return

	# start timer for total execution
	start_time = time.time()

	frame_idx = 0
	frame_annotations = []  # list indexed by frame_idx, each is dict id->bbox

	# instantiate tracker before first pass
	tracker = SortTracker(max_disappeared=args.max_disappeared,
	                      max_distance=args.max_distance,
	                      min_track_area=args.min_track_area,
	                      min_track_side=(args.min_track_side if args.min_track_side > 0 else None),
	                      new_iou_gate=args.new_iou_gate)

	# prepare temp folder to save per-ID crops (use local ./temp/extractions)
	# previously used ./temp; now store in ./temp/extractions
	base_temp = os.path.abspath(os.path.join(".", "temp", "extractions"))
	os.makedirs(base_temp, exist_ok=True)
	save_counts = {}  # id -> saved image count

	# optional writer for first pass (created on first successful read to get frame size & fps)
	writer_first = None
	fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

	# First pass: run tracker and store per-frame assignments
	while True:
		ret, frame = cap.read()
		if not ret:
			break

		# create writer_first if requested and not yet created
		if args.save and writer_first is None:
			h, w = frame.shape[:2]
			fourcc = cv2.VideoWriter_fourcc(*"mp4v")
			writer_first = cv2.VideoWriter(args.save, fourcc, fps, (w, h))

		detections = detect_rectangles(frame, min_area=args.min_area, thresh_val=args.thresh,
		                               iou_thresh=args.iou, use_watershed=args.watershed,
		                               large_area_factor=args.large_area_factor,
		                               watershed_bottom_margin=args.watershed_bottom_margin)
		objects = tracker.update(detections, frame_idx=frame_idx)

		# --- NOTE: removed saving crops here (we only collect annotations). ---
		# store a shallow copy of objects for this frame
		frame_annotations.append(dict(objects))

		# write annotated frame if requested
		if writer_first is not None:
			out = tracker.annotate(frame, objects)
			writer_first.write(out)

		frame_idx += 1

	# finalize trajectories and compute valid IDs
	tracker.finalize()
	valid_ids = tracker.filter_trajectories(min_length=args.min_track_length)

	# ---------- new: extract crops from the REAL color video using stored annotations ----------
	# choose color source (fallback to args.video if not provided)
	color_src = args.color if args.color else args.video
	color_cap = open_capture(color_src)
	if not color_cap.isOpened():
		print("Warning: impossible d'ouvrir la vidéo couleur pour l'extraction des crops:", color_src)
	else:
		# iterate frames of color video and save crops according to frame_annotations
		i = 0
		while True:
			ret_c, frame_c = color_cap.read()
			if not ret_c:
				break
			ann = frame_annotations[i] if i < len(frame_annotations) else {}
			if ann:
				fh_c, fw_c = frame_c.shape[:2]
				for oid, bbox in ann.items():
					x1, y1, x2, y2 = bbox
					x1 = max(0, int(x1)); y1 = max(0, int(y1))
					x2 = min(fw_c, int(x2)); y2 = min(fh_c, int(y2))
					if x2 <= x1 or y2 <= y1:
						continue
					crop = frame_c[y1:y2, x1:x2]
					oid_dir = os.path.join(base_temp, str(oid))
					os.makedirs(oid_dir, exist_ok=True)
					cnt = save_counts.get(oid, 0)
					filename = f"{i:06d}.png"
					fullpath = os.path.join(oid_dir, filename)
					if os.path.exists(fullpath):
						filename = f"{i:06d}_{cnt:03d}.png"
						fullpath = os.path.join(oid_dir, filename)
					cv2.imwrite(fullpath, crop)
					save_counts[oid] = cnt + 1
			i += 1
		color_cap.release()
	# ------------------------------------------------------------------------------------------

	# print runtime (remove 'valid' count)
	duration = time.time() - start_time
	print(f"Total trajectories: {len(tracker.completed)}, runtime: {duration:.2f}s")

	# warn if no video is being saved
	if (not args.save) and (not args.save_validated):
		print("Attention: aucune vidéo enregistrée (utilisez --save et/ou --save_validated pour sauvegarder)")

	# Second pass: replay original frames and display/save only validated tracks
	if writer_first is not None:
		writer_first.release()
	cap.release()

	cap = open_capture(args.video)
	if not cap.isOpened():
		print("Erreur: impossible d'ouvrir la source pour la seconde passe", args.video)
		return

	ret, sample = cap.read()
	if not ret:
		print("Erreur: replay impossible, vidéo vide")
		cap.release()
		return
	h, w = sample.shape[:2]
	writer_valid = None
	if args.save_validated:
		fourcc = cv2.VideoWriter_fourcc(*"mp4v")
		writer_valid = cv2.VideoWriter(args.save_validated, fourcc, fps, (w, h))

	i = 0
	# Second pass: replay original frames and write only validated tracks (no GUI)
	while True:
		ret, frame = cap.read()
		if not ret:
			break
		ann = frame_annotations[i] if i < len(frame_annotations) else {}
		# keep only validated ids
		ann_valid = {oid: bbox for oid, bbox in ann.items() if oid in valid_ids}
		out = tracker.annotate(frame, ann_valid)
		if writer_valid is not None:
			writer_valid.write(out)
		# no cv2.imshow / cv2.waitKey anymore
		i += 1

	cap.release()
	if writer_valid is not None:
		writer_valid.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()