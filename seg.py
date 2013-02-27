
from PIL import Image
import numpy as np
import sklearn.svm as svm
import pickle, random

def DetectGreen(labels):
	labelsBinary = np.zeros((labels.shape[0],labels.shape[1]), dtype = np.uint8)

	green = [197, 238, 131]
	greenThresh = 10.

	for x in range(labels.shape[1]):
		print x, labels.shape[1]
		for y in range(labels.shape[0]):
			col = labels[y,x,:]
			distGreen = np.power(green - col, 2.).sum() ** 0.5
			labelsBinary[y,x] = (distGreen < greenThresh) * 255
	return labelsBinary

if __name__=="__main__":

	if 0:
		labels = np.array(Image.open("su10nw-true.png").convert("RGB"))
		labelsBinary = DetectGreen(labels)

		labelsBinaryImg = Image.fromarray(labelsBinary)
		labelsBinaryImg.save("binlabels.png")
		labelsBinaryImg.show()

	if 0:
		features = np.array(Image.open("su10nw.png").convert("RGB"))
		greenBinary = DetectGreen(features)

		greenBinaryImg = Image.fromarray(greenBinary)
		greenBinaryImg.save("greenpix.png")
		greenBinaryImg.show()

	features = np.array(Image.open("su10nw.png").convert("RGB"))
	labelsBinary = np.array(Image.open("binlabels.png").convert("L"))
	greenBinaryImg = np.array(Image.open("greenpix.png").convert("L"))

	if 0:
		#Locate interesting non-green but positive pixels
		interestingTrainPos = []
		for x in range(20, labelsBinary.shape[1]-20):
			print x, labelsBinary.shape[1], len(interestingTrainPos)
			for y in range(20, labelsBinary.shape[0]-20):
				if labelsBinary[y,x] > 128 and greenBinaryImg[y,x] < 128:
					#print x, y, labelsBinary[y,x], greenBinaryImg[y,x]
					interestingTrainPos.append((x, y))

		#Generate random training positions
		randTrainPos = []
		for count in range(len(interestingTrainPos)):
			px= np.random.randint(20, labelsBinary.shape[1]-20)
			py= np.random.randint(20, labelsBinary.shape[0]-20)
			randTrainPos.append((px, py))

		pickle.dump(randTrainPos, open("randTrainPos.dat","wb"), protocol = -1)
		pickle.dump(interestingTrainPos, open("interestingTrainPos.dat","wb"), protocol = -1)


	randTrainPos = pickle.load(open("randTrainPos.dat","rb"))
	interestingTrainPos = pickle.load(open("interestingTrainPos.dat","rb"))
	combinedTrainPos = random.sample(interestingTrainPos, 10000)
	combinedTrainPos.extend(random.sample(randTrainPos, 10000))

	trainData = []
	trainLabels = []	
	for px, py in combinedTrainPos:
		patch = features[py-10:py+10, px-10:px+10, :]
		patchVector = patch.reshape(patch.size)
		label = labelsBinary[py, px] > 128
		#patchImg = Image.fromarray(patch)
		#patchImg.show()
		#print label

		trainData.append(patchVector)
		trainLabels.append(label)

	print "Whitening"
	trainData = np.array(trainData)
	print trainData.shape
	trainDataMean = trainData.mean(axis=0)
	print trainDataMean.shape
	trainDataVar = trainData.var(axis=0)
	print trainDataVar.shape
	if 0:
		pickle.dump((trainDataMean, trainDataVar), open("scaling.dat","wb"), protocol = -1)
	if 1:
		(trainDataMean, trainDataVar) = pickle.load(open("scaling.dat","rb"))

	a = np.power(trainDataVar, 0.5)
	whitenedTrainData = (trainData - trainDataMean) / a

	if 0:

		print "Training classifier"

		cl = svm.SVC()
		cl.fit(whitenedTrainData, trainLabels)
		pickle.dump(cl, open("class.dat","wb"), protocol = -1)
	
	if 0:
		testSampleInd = random.sample(range(len(trainLabels)), 1000)
		whitenedTrainData = whitenedTrainData[testSampleInd, :]
		trainLabels = np.array(trainLabels)[testSampleInd]

		print "Test classifier"
		cl = pickle.load(open("class.dat","rb"))
		pred = cl.predict(whitenedTrainData)

		match = 0
		for p, tr in zip(pred, trainLabels):
			match += (p == tr)
		print float(match) / len(trainLabels)

	if 1:
		cl = pickle.load(open("class.dat","rb"))
		labelsBinary = np.zeros((features.shape[0],features.shape[1]), dtype = np.uint8)

		#Process image
		for px in range(20,features.shape[1]-20):
			print px
			for py in range(20,features.shape[0]-20):
				patch = features[py-10:py+10, px-10:px+10, :]
				patchVector = patch.reshape(patch.size)
				whitenedData = (patchVector - trainDataMean) / a
				pred = cl.predict(whitenedData)
				#print px,py, pred
				labelsBinary[py,px] = 255 * int(pred[0])

			labelsBinaryImg = Image.fromarray(labelsBinary)
			labelsBinaryImg.save("pred.png")


