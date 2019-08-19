import math
import os

#todo (currently in qr.cpp)
def poly_fit(in_pts, in_stds, name):
    f = open(name+'_pts','w')
    f.write(' '.join([str(i) for i in in_pts]))
    f.close()
    f = open(name+'_stds','w')
    f.write(' '.join([str(i) for i in in_stds]))
    f.close()
    os.system('./utils/poly/qr '+name+' > '+name+'_out')
    out = open(name+'_out','r').read().split()
    out = [float(i) for i in out]
    # print(out)
    return out

def softplus(x):
    return math.log(1 + math.exp(x))

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class PathData():
	def __init__(self):
		self.points = []
		self.prob = 0.0
		self.std = 0.0
		self.stds = []
		self.poly = []

class LeadData():
	def __init__(self):
		self.dist = 0.0
		self.prob = 0.0
		self.std = 0.0
		self.rel_v = 0.0
		self.rel_v_std = 0.0

class PathModel():
	def __init__(self,data=None):
		self.path = PathData()
		self.right_lane = PathData()
		self.left_lane = PathData()
		self.lead = LeadData()

		if data is not None:
			self.load_data(data)

	def load_data(self,data):
		path_len = 100
		poly_deg = 4
		out = {
			'path':data[0 : path_len*2],
			'left_lane':data[path_len*2 : path_len*2 + path_len*2 + 1],
			'right_lane':data[path_len*2 + path_len*2 + 1 : path_len*2 + (path_len*2 + 1)*2],
			'lead':data[path_len*2+(path_len*2+1)*2:]
		}
		for i in range(path_len):
			self.path.points.append(out['path'][i])
			self.left_lane.points.append(out['left_lane'][i]+1.8)
			self.right_lane.points.append(out['right_lane'][i]-1.8)
			self.path.stds.append(softplus(out['path'][path_len+i]))
			self.left_lane.stds.append(softplus(out['left_lane'][path_len+i]))
			self.right_lane.stds.append(softplus(out['right_lane'][path_len+i]))

		self.path.std = softplus(out['path'][int(path_len+path_len/2)])
		self.left_lane.std = softplus(out['left_lane'][int(path_len+path_len/2)])
		self.right_lane.std = softplus(out['right_lane'][int(path_len+path_len/2)])

		self.path.prob = 1.0
		self.left_lane.prob = sigmoid(out['left_lane'][path_len*2])
		self.right_lane.prob = sigmoid(out['right_lane'][path_len*2])

		self.path.poly = poly_fit(self.path.points,self.path.stds,'utils/poly/path')
		self.left_lane.poly = poly_fit(self.left_lane.points,self.left_lane.stds,'utils/poly/left')
		self.right_lane.poly = poly_fit(self.right_lane.points,self.right_lane.stds,'utils/poly/right')

		mdn_max_idx = 0
		lead_mdn_n=5
		for i in range(1,lead_mdn_n):
			if out['lead'][i*5+4] > out['lead'][mdn_max_idx*5+4]:
				mdn_max_idx = i

		for i in range(path_len):
			self.path.points[i] = self.path.poly[0] * (i*i*i) \
									+ self.path.poly[1] * (i*i) \
									+ self.path.poly[2] * (i) \
									+ self.path.poly[3]
			self.left_lane.points[i] = self.left_lane.poly[0] * (i*i*i) \
									+ self.left_lane.poly[1] * (i*i) \
									+ self.left_lane.poly[2] * (i) \
									+ self.left_lane.poly[3]
			self.right_lane.points[i] = self.right_lane.poly[0] * (i*i*i) \
									+ self.right_lane.poly[1] * (i*i) \
									+ self.right_lane.poly[2] * (i) \
									+ self.right_lane.poly[3]

	def __repr__(self):
		#todo
		res = 'PathModel!!'
		return res
