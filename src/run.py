import os
import numpy as np
import datetime

#initial rnn state
a = [0.0]*512
a = np.array(a).astype(np.float32).reshape(1,512)
a.tofile('data/states.raw')
n = datetime.datetime.now()


for i in range(50):
	with open('data/raw_list.txt','w') as f:
		f.write('vision_lambda/div:0:=/mnt/d/stuff/openpilot_test/data/raws/'+str(i)+'.raw rnn_state:0:=/mnt/d/stuff/openpilot_test/data/states.raw')
	os.system('snpe-net-run --container driving_model.dlc --input_list data/raw_list.txt > out.txt')
	os.system('mv /output/Result_0/outputs/concat:0.raw data/results/resRNN'+str(i)+'.raw')
	
	#save output RNN state or not
	RNN = True
	if RNN:
		out = np.fromfile('data/results/resRNN'+str(i)+'.raw',dtype=np.float32)
		out = np.array(out)
		new_rnn = out[-512:].astype(np.float32).reshape(1,512)

		new_rnn.tofile('data/states.raw')
print(datetime.datetime.now()-n)