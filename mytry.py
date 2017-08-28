import logRegres
from numpy import *

dataArr,labelMat = logRegres.loadDataSet()
# print(dataArr)
# print(labelMat)
# print(logRegres.gradAscent(dataArr, labelMat))

# 批梯度上升算法
# weights = logRegres.gradAscent(dataArr, labelMat)
# logRegres.plotBestFit(weights.getA())

# 随机梯度上升算法
weights = logRegres.stocGradAscent0(array(dataArr), labelMat)
logRegres.plotBestFit(weights)

# dataMat,labelMat=logRegres.loadDataSet()
# weights = logRegres.stocGradAscent1(array(dataMat),labelMat)
# logRegres.plotBestFit(weights)

logRegres.multiTest()
