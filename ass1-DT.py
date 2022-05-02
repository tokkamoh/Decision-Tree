import pandas as pn
from sklearn import preprocessing
import matplotlib as plt
from sklearn import tree
import statistics
import random

house_votes = pn.read_csv(r"E:\Bioninformatics\year4,sem1\Machine Learning and Bioinformatics\Assignments\house-votes-84.data", header=None)
copy = house_votes
outputY = house_votes[0]

features = copy.drop(columns=copy.columns[0])

modeCol = features.mode()
modeColStr = str(modeCol)
modeColSplit = modeColStr.split()
modeLast = modeColSplit[17:]
# print(modeCol, "\n", modeColStr, "\n", modeColSplit, "\n", modeLast)
i = 0

process = preprocessing.LabelEncoder()
outProcess = preprocessing.LabelEncoder()
for x in features:
    features[x] = features[x].replace({"?": modeLast[i]})
    process.fit(features[x])
    features[x] = process.transform(features[x])
    i = i + 1

outProcess.fit(outputY)
outputY = outProcess.transform(outputY)

dt = tree.DecisionTreeClassifier()

rangeList = [30, 40, 50, 60, 70, 80]
accuracy = []
sizList = []

accuracyX = []
sizListX = []
# Random 25% splits
for x in range(3):
    counterX = 0
    split = random.randint(1, int(len(features) - (len(features)*0.25)))
    siz = int(split + len(features) * 0.25)
    # print(split, siz, len(outputY))
    dt = dt.fit(features[split:siz], outputY[split:siz])
    outA = dt.predict(features[siz:])
    outB = dt.predict(features[:split])
    for x in range(len(outputY[siz:])):
        if siz == len(outputY)-1:
            # print("BROKE")
            break
        elif outA[x] == outputY[siz:][x]:
            counterX += 1
    for x in range(len(outputY[:split])):
        if split == 0:
            # print("0, break")
            break
        if outB[x] == outputY[:split][x]:
            counterX += 1
    accuracyX.append((counterX / (len(outputY[siz:]) + len(outputY[:split])) * 100))
    nodeSize = dt.tree_.node_count
    sizListX.append(nodeSize)

print(accuracyX)

for i in rangeList:
    counter = 0
    siz = int((i / 100) * len(features))
    dt = dt.fit(features[:siz], outputY[:siz])
    outA = dt.predict(features[siz:])
    for x in range(len(outputY[siz:])):
        if outA[x] == outputY[siz:][x]:
            counter += 1
    accuracy.append((counter / len(outputY[siz:])) * 100)
    siz = dt.tree_.node_count
    sizList.append(siz)

meanSize = statistics.mean(sizList)
minSize = min(sizList)
maxSize = max(sizList)
print("Mean of the tree size with different train sizes: ", meanSize)
print("Min size of the nodes in the tree with different train sizes: ", minSize)
print("Max size of the nodes in the tree with different train sizes: ", maxSize)
print("Accuracy starting from training size 30 till 80: ", accuracy)
index = accuracy.index(max(accuracy))
dt = dt.fit(features[:rangeList[index]], outputY[:rangeList[index]])
outA = dt.predict(features[rangeList[index]:])

plt.pyplot.plot(rangeList, sizList, color='black', linewidth=1, marker='o', markerfacecolor='pink', markersize=8)
plt.pyplot.show()

plt.pyplot.plot(rangeList, accuracy, color='black', linewidth=1, marker='o', markerfacecolor='purple', markersize=8)
plt.pyplot.show()

tree.plot_tree(dt, filled=True)
# fig = plt.pyplot.figure(figsize=(25, 20))
plt.pyplot.show()
