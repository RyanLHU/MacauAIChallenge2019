import os, cv2, re, random
import numpy as np
from keras.models import load_model


def predict(img_path=img_path):
	model = load_model('sign.h5')
	x = (cv2.resize(cv2.imread(img_path), (150,150), interpolation=cv2.INTER_CUBIC))
	x = x.reshape(1,150,150,3)
	y = model.predict(np.array(x))

	y = list(y[0])
	max_value = max(y)
	label = y.index(max_value)
	return label