import cv2
import numpy as np

MODEL_PATH_MAX_VERTICES_CNT=98

eon_focal_length=910
medmodel_zoom = 1.
# MEDMODEL_INPUT_SIZE = (512, 256)
# MEDMODEL_YUV_SIZE = (MEDMODEL_INPUT_SIZE[0], MEDMODEL_INPUT_SIZE[1] * 3 // 2)
# MEDMODEL_CY = 47.6
# intrinsic = np.array([
# 	[ eon_focal_length / medmodel_zoom, 0., 0.5 * MEDMODEL_INPUT_SIZE[0]],
# 	[0., eon_focal_length / medmodel_zoom, MEDMODEL_CY],
# 	[0., 0., 1.]
# ])

intrinsic = np.array([
  [910., 0., 582.],
  [0., 910., 437.],
  [0.,   0.,   1.]
])

def car_space_to_full_frame(car_space_projective):
	#?
	# extrinsic = np.array([[ 9.86890774e-03, -9.99951124e-01, -6.39820937e-04, 0.00000000e+00],
	#    [-6.46961704e-02, -5.42101086e-20, -9.97905016e-01, 1.22000003e+00],
	#    [ 9.97856200e-01,  9.88962594e-03, -6.46930113e-02, 0.00000000e+00]])
	#27
	extrinsic = np.array([
		[-3.65592428e-02,  9.99330759e-01,  1.20524236e-03, -1.44629076e-03],
		[ 4.35471088e-02,  3.88210319e-04,  9.99051273e-01, -1.19886160e+00],
		[-9.98382211e-01, -3.65770459e-02,  4.35321592e-02, -5.22385910e-02]])
	#34
	# extrinsic = np.array([
	# 	[ 3.51227634e-02, -9.99382317e-01, -1.19575066e-03, 0.00000000e+00],
    # 	[-3.40251774e-02,  0.00000000e+00, -9.99421000e-01, 1.22000003e+00],
    # 	[ 9.98803616e-01,  3.51431146e-02, -3.40041593e-02, 0.00000000e+00]
	# ])

	ep = extrinsic.dot(car_space_projective)
	# print(ep.shape)
	kep = intrinsic.dot(ep)
	p_image = np.array([kep[0]/kep[2], kep[1]/kep[2], 1])
	return p_image

class Pvd():
	class pt():
		def __init__(self,x,y):
			self.x = x
			self.y = y
		def __repr__(self):
			return '('+str(self.x)+','+str(self.y)+')'
	def __init__(self):
		self.cnt = 0
		self.v = []
	def add_pt(self,x,y):
		p = self.pt(x,y)
		self.v.append(p)
	def __repr__(self):
		return str(self.v)

def update_lane_line_data(points, off, is_ghost):
	pvd = Pvd()
	for i in range(MODEL_PATH_MAX_VERTICES_CNT // 2):
		px = float(i)
		py = points[i] - off
		p_car_space = np.array([px, py, 0, 1])
		p_full_frame = car_space_to_full_frame(p_car_space)
		x = p_full_frame[0]
		y = p_full_frame[1]
		if x<0 or y<0:
			continue
		pvd.add_pt(x,y)
		pvd.cnt += 1
	for i in range(MODEL_PATH_MAX_VERTICES_CNT // 2, 0, -1):
		px = float(i)
		if is_ghost:
			py = points[i]-off
		else:
			py = points[i]+off
		p_car_space = np.array([px, py, 0, 1])
		p_full_frame = car_space_to_full_frame(p_car_space)
		x = p_full_frame[0]
		y = p_full_frame[1]
		if x<0 or y<0:
			continue
		pvd.add_pt(x,y)
		pvd.cnt += 1
	return pvd

def update(p):
	p1 = update_lane_line_data(p.points,0.025*p.prob, False)
	var = min(p.std, 0.7)
	p2 = update_lane_line_data(p.points,-var, True)
	p3 = update_lane_line_data(p.points,var, True)
	return p1,p2,p3

def draw_lane(img, model):
	p1,p2,p3 = update(model.path)
	lp1,lp2,lp3 = update(model.left_lane)
	rp1,rp2,rp3 = update(model.right_lane)
	img = draw(img, p1, (255,0,0), 10)
	# img = draw(img, p2, (255,0,0), 2)
	# img = draw(img, p3, (255,0,0), 2)
	img = draw(img, lp1, (0,255,0), 5)
	# img = draw(img, lp2, (0,255,0), 1)
	# img = draw(img, lp3, (0,255,0), 1)
	img = draw(img, rp1, (0,0,255), 5)
	# img = draw(img, rp2, (0,0,255), 1)
	# img = draw(img, rp3, (0,0,255), 1)

	return img

def draw(img, line, c, w):
	l = line.v[1:50]
	# print(l)
	for j in range(1,len(l)):
		pt1 = (int(l[j-1].x),int(l[j-1].y))
		pt2 = (int(l[j].x),int(l[j].y))
		cv2.line(img, pt1, pt2, c, w)
	return img