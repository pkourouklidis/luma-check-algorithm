from PIL import Image
from scipy import stats
import numpy as np

def detector(trainSet, liveSet, parameters):
    firstColumnTrain = trainSet.axes[1][0]
    firstColumnLive = liveSet.axes[1][0]

    luma_train = []
    for array in trainSet[firstColumnTrain]:
        image = Image.fromarray((np.reshape(array, (224,224,3)) * 255).astype("uint8"))
        grayscale = image.convert("L")
        luma_train.append(np.mean(np.asarray(grayscale)))
    
    luma_live = []
    for array in liveSet[firstColumnLive]:
        image = Image.fromarray((np.reshape(array, (224,224,3)) * 255).astype("uint8"))
        grayscale = image.convert("L")
        luma_live.append(np.mean(np.asarray(grayscale)))
    
    pValue = stats.ks_2samp(luma_train, luma_live)[1]
    threshold = float (parameters.get("pValue", 0.05))
    return int(pValue < threshold), pValue