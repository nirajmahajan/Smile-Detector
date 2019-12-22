from helpers import *
from keras.utils import plot_model

parser = argparse.ArgumentParser(description='Image Detection')
parser.add_argument('-use_trained_model', action = 'store_true')
args = parser.parse_args()

[train_x, train_y, test_x, test_y, classes] = loadDataset()

if(not args.use_trained_model):
	myModel = MyModel(train_x.shape[1:])
	myModel.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
	myModel.fit(x = train_x, y = train_y, epochs = 40, batch_size = 16)

	preds = myModel.evaluate(test_x, test_y, batch_size=32, verbose=1, sample_weight=None)
	print("\nResults after training:\nLoss = {}".format(preds[0]))
	print("Test Accuracy = {}".format(preds[1]))
	myModel.save('model/trained_model.h5')
	plot_model(myModel, to_file='model/trained_model.png')
else:
	myModel = load_model('model/trained_model.h5')
	preds = myModel.evaluate(test_x, test_y, batch_size=32, verbose=1, sample_weight=None)
	print("\nLoss = {}".format(preds[0]))
	print("Test Accuracy = {}".format(preds[1]))