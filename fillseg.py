from PIL import Image
import pickle, random
import numpy as np
import sklearn.cluster as cluster
import sklearn.svm as svm
import matplotlib.pyplot as plt

OSGREEN = (197, 238, 131)
OSLIGHTGREY = (230, 230, 230)
OSDARKBLUE = (74, 131, 246)
OSLIGHTYELLOW = (255, 255, 238)
OSDARKGREY = (90, 90, 90)

def ExtractAllowedColours(tile, tileTrue):
	trueCols = {}
	falseCols = {}
	tilel = tile.load()
	tileTruel = tileTrue.load()

	#Count colour pixels
	for x in range(tile.size[0]):
		print x, tile.size[0], len(trueCols), len(falseCols)
		for y in range(tile.size[1]):

			tc = OSGREEN
			diff = ((tileTruel[x,y][0] - tc[0]) ** 2. \
				+ (tileTruel[x,y][1] - tc[1]) ** 2.
				+ (tileTruel[x,y][2] - tc[2]) ** 2.) ** 0.5
			tr = (diff < 10.)

			if tr:
				if tilel[x,y] not in trueCols:
					trueCols[tilel[x,y]] = 0
				trueCols[tilel[x,y]]+=1
			else:
				if tilel[x,y] not in falseCols:
					falseCols[tilel[x,y]] = 0
				falseCols[tilel[x,y]]+=1

	#Normalise
	total = 0
	for v in trueCols.values(): total += v
	print "num trueCols",total,len(trueCols)
	for col in trueCols:
		trueCols[col] /= float(total)

	total = 0
	for v in falseCols.values(): total += v
	print "num falseCols",total,len(falseCols)
	for col in falseCols:
		falseCols[col] /= float(total)

	return trueCols, falseCols

def CheckColIsCandidate(col, clust, trueColsClust, falseColsClust):

	ind = clust.predict(map(float,col))
	trueFrac = trueColsClust[ind]
	falseFrac = falseColsClust[ind]
	if falseFrac == 0.:
		assert trueFrac > 0.
		return True

	ratio = trueFrac / falseFrac
	thres = 0.05
	return ratio > thres

def FillOfColourModel(img, clust, trueColsClust, falseColsClust, pos, mask):
	imgl = img.load()
	col = imgl[pos[0],pos[1]]
	assert CheckColIsCandidate(col, clust, trueColsClust, falseColsClust)

	posBuff = [pos]
	mask[pos[1], pos[0]] = 1
	adjPosSet = [(0,1),(0,-1),(-1,0),(1,0)]
	count = 0

	while(len(posBuff)>0):
		currentPos = posBuff.pop(0)
		if count % 1000 == 0:
			print currentPos, len(posBuff), count
		assert mask[currentPos[1], currentPos[0]]

		for adjPos in adjPosSet:
			offPos = (currentPos[0] + adjPos[0],currentPos[1] + adjPos[1])
			if offPos[0] < 0: continue
			if offPos[0] >= img.size[0]: continue
			if offPos[1] < 0: continue
			if offPos[1] >= img.size[1]: continue
			if mask[offPos[1], offPos[0]]: continue

			col = imgl[offPos]
			isCand = CheckColIsCandidate(col, clust, trueColsClust, falseColsClust)
			if isCand:
				mask[offPos[1], offPos[0]] = 255
				if offPos not in posBuff: posBuff.append(offPos)
		
		count+=1
		#if count % 1000 == 0:
		#	im = Image.fromarray(out)
		#	im.save(str(count)+".png")


def AdjPixMatch(pos, col, img, imgl):
	adjPosSet = [(-1,0),(0,1),(0,-1),(1,0)]
	for adjPos in adjPosSet:
		offPos = (pos[0] + adjPos[0],pos[1] + adjPos[1])
		if offPos[0] < 0: continue
		if offPos[0] >= img.size[0]: continue
		if offPos[1] < 0: continue
		if offPos[1] >= img.size[1]: continue

		colPix = imgl[offPos]
		if colPix == col: return True
		
	return False

def SimpleFill(img, pos, targetCol, setCol):
	imgl = img.load()
	found = 1
	assert imgl[pos] == targetCol
	imgl[pos] = setCol
	countIter = 0

	while(found):
		found = 0
		#Left to right, top to bottom
		for x in range(img.size[0]):
			#print x, img.size[0], found
			for y in range(img.size[1]):
				if imgl[x,y] == targetCol and AdjPixMatch((x,y), setCol, img, imgl):
					imgl[x,y] = setCol
					found += 1

		print "Left to right, top to bottom, found", found
		if found == 0: continue
		found = 0

		#Top to bottom, right to left
		for y in range(img.size[1]):
			#print y, img.size[1], found
			for x in range(img.size[0]-1,-1,-1):
				if imgl[x,y] == targetCol and AdjPixMatch((x,y), setCol, img, imgl):
					imgl[x,y] = setCol
					found += 1

		print "Left to right, buttom to top, found", found
		if found == 0: continue
		found = 0

		#Left to right, buttom to top
		for x in range(img.size[0]):
			#print x, img.size[0], found
			for y in range(img.size[1]-1,-1,-1):
				if imgl[x,y] == targetCol and AdjPixMatch((x,y), setCol, img, imgl):
					imgl[x,y] = setCol
					found += 1

		print "Top to bottom, Left to right, found", found
		if found == 0: continue
		found = 0

		#Bottom to top, right to left
		for y in range(img.size[1]-1,-1,-1):
			#print y, img.size[1], found
			for x in range(img.size[1]-1,-1,-1):
				if imgl[x,y] == targetCol and AdjPixMatch((x,y), setCol, img, imgl):
					imgl[x,y] = setCol
					found += 1

		print "Top to bottom, right to left, found", found
		
		countIter += 1
		img.save("iter"+str(countIter)+".png")

def PatchToHist(img, pos, clusters):
	patch = img.crop((pos[0]-10,pos[1]-10,pos[0]+10,pos[1]+10))
	patchl = patch.load()
	freq = [0 for i in range(len(clusters.cluster_centers_))]

	for x in range(0, patch.size[0], 2):
		for y in range(0, patch.size[1], 2):
			ind = clusters.predict(map(float,patchl[x,y]))[0]
			#print x, y, imgl[x,y], ind
			freq[ind] += 1

	return freq

def CalcDataScale(data):
	trainData = np.array(data)
	trainDataMean = trainData.mean(axis=0)
	trainDataVar = trainData.var(axis=0)
	trainDataVar = trainDataVar + (trainDataVar == 0.) #Prevent div by zero
	return trainDataMean, trainDataVar
	#a = np.power(trainDataVar, 0.5)
	#whitenedTrainData = (trainData - trainDataMean) / a

if __name__=="__main__":
	tile = Image.open("su10nw.png")
	tile = tile.convert("RGB")
	tilel = tile.load()
	tileTrue = Image.open("su10nw-true.png")
	tileTrue = tileTrue.convert("RGB")
	tileTruel = tileTrue.load()

	#Extract allowed wood colours
	if 0:
		trueCols, falseCols = ExtractAllowedColours(tile, tileTrue)
		pickle.dump(trueCols, open("trueCols.dat","wb"), protocol=-1)
		pickle.dump(falseCols, open("falseCols.dat","wb"), protocol=-1)
	else:
		trueCols = pickle.load(open("trueCols.dat","rb"))
		falseCols = pickle.load(open("falseCols.dat","rb"))

	#print trueCols[OSDARKGREY]
	#print falseCols[OSDARKGREY]
	#print trueCols[OSDARKGREY] / falseCols[OSDARKGREY]
	
	if 1:
		allCols = {}
		for col in trueCols:
			allCols[col] = trueCols[col]
		for col in falseCols:
			if col not in allCols: allCols[col] = 0
			allCols[col] += falseCols[col]

	if 0:
		kmeans = cluster.KMeans(50)
		clust = kmeans.fit(np.array(allCols.keys(), dtype=np.float))
		trueColsClust, falseColsClust = [0. for i in range(50)], [0. for i in range(50)]

		for col in trueCols:
			ind = clust.predict(np.array(col, dtype=np.float))[0]
			trueColsClust[ind] += trueCols[col]

		for col in falseCols:
			ind = clust.predict(np.array(col, dtype=np.float))[0]
			falseColsClust[ind] += falseCols[col]

		pickle.dump((clust, trueColsClust, falseColsClust), open("colModel.dat","wb"), protocol=-1)
	else: 
		clust, trueColsClust, falseColsClust = pickle.load(open("colModel.dat","rb"))

	if 0:
		mask = np.zeros((tile.size[1],tile.size[0]), dtype=np.uint8)
		candIterCount = 0
		for x in range(tile.size[0]):
			for y in range(tile.size[0]):
				if tilel[x,y] != OSGREEN or mask[y,x] != 0: continue
				FillOfColourModel(tile, clust, trueColsClust, falseColsClust, (x, y), mask)

				candIterCount += 1
				#maskIm = Image.fromarray(mask)
				#maskIm.save("candit"+str(candIterCount)+".png")

		pickle.dump(mask, open("mask.dat","wb"), protocol=-1)
	else:
		mask = pickle.load(open("mask.dat","rb"))

	if 1:
		kmeans = cluster.KMeans(10)
		modelClust = kmeans.fit(np.array(allCols.keys(), dtype=np.float))

	if 0:
		#Create updating model colour clustering
		keyind = modelClust.predict(map(float,OSLIGHTGREY))

		#Find examples to train a model for removing excess roads
		maskIm = Image.fromarray(mask)
		maskIml = maskIm.load()
		pos, neg = [], []
		for x in range(maskIm.size[0]):	
			for y in range(maskIm.size[1]):
				if maskIml[x,y] != 255: continue
				col = tilel[x,y]
				ind = modelClust.predict(map(float,col))
				if ind != keyind: continue

				tc = OSGREEN
				diff = ((tileTruel[x,y][0] - tc[0]) ** 2. \
					+ (tileTruel[x,y][1] - tc[1]) ** 2.
					+ (tileTruel[x,y][2] - tc[2]) ** 2.) ** 0.5
				tr = (diff < 10.)

				if tr: pos.append((x,y))
				else: neg.append((x,y))
				#print x,y,col,tr,len(pos),len(neg)
			print x,len(pos),len(neg)
			
		pickle.dump((pos, neg), open("posneg.dat","wb"), protocol=-1)
	else:
		pos, neg = pickle.load(open("posneg.dat","rb"))

	print len(pos), len(neg)
	posSamp = random.sample(pos, 1000)
	negSamp = random.sample(neg, 1000)
	featureData, labelData = [], []

	print "Extract training patches"
	for posn in posSamp:
		hist = PatchToHist(tile, posn, modelClust)
		featureData.append(hist)
		labelData.append(1)

	for posn in negSamp:
		hist = PatchToHist(tile, posn, modelClust)
		featureData.append(hist)
		labelData.append(0)

	print "Calc scales"
	dataMean, dataVar = CalcDataScale(featureData)
	dataStd = np.power(dataVar, 0.5)
	whFeatureData = (featureData - dataMean) / dataStd

	print "Train classifier"
	classifier = svm.SVC()
	classifier.fit(whFeatureData, labelData)
	
	pickle.dump((modelClust, dataMean, dataVar, classifier), open("removemodel.dat","wb"), protocol=-1)

	#Test classifier
	print "Test classifier"
	posSamp = random.sample(pos, 1000)
	negSamp = random.sample(neg, 1000)
	featureData, labelData = [], []

	for posn in posSamp:
		hist = PatchToHist(tile, posn, modelClust)
		featureData.append(hist)
		labelData.append(1)

	for posn in negSamp:
		hist = PatchToHist(tile, posn, modelClust)
		featureData.append(hist)
		labelData.append(0)

	whFeatureData = (featureData - dataMean) / dataStd
	pred = classifier.predict(whFeatureData)
	confm = np.zeros((2,2))	
	for pr, tr in zip(pred, labelData):
		confm[pr,tr] += 1

	print "Confusion matrix"
	print confm

