import scipy
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

parser = argparse.ArgumentParser(description='Sign Detection')
parser.add_argument('--path', required = True)

args = vars(parser.parse_args())

if(not os.path.exists(args['path'])):
	print('Not a valid path')
	os._exit(1)

if(not os.path.isfile(args['path'])):
	print('Not a file')
	os._exit(1)
if(not os.path.isfile('model/trained_model.h5'):
	print('Train the model first')
	os._exit(1)

fname = args['path']
img = image.load_img(fname, target_size=(64, 64))
imshow(img)

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

myModel = load_model('model/trained_model.h5')
print(myModel.predict(x))
plt.show()