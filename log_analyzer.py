# Parse learning curve data(loss, accuracy) from keras log
#
# Usage:
# radioactive_area.py > log.txt
# log_analyzer.py log.txt > learning_curve.csv

import sys
argvs = sys.argv
if len(argvs)<2:
	sys.exit()

filename = argvs[1] # 'log_simple_sigmoid_50000.txt'
flg = False
loss = 0
acc = 0
print('epoch,loss,acc')
for line in open(filename, 'r'):
	fields = line.split()
	if len(fields) > 1:
		f1 = fields[1]
		if  len(fields) > 3 and 'ETA' in fields[3]:
			flg = True
		if len(fields) > 10 and flg and ('step' in fields[4]):
			loss = fields[7]
			acc  = fields[10]
		if 'Epoch' in fields[0]:
			ep = f1.split('/')[0]
			print("{},{},{}".format(ep, loss, acc))
			flg = False
