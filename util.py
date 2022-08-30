import cv2
import tensorflow as tf
import skimage
import skimage.io
import skimage.viewer
import warnings
import numpy as np

#importing gesture icons
classimages0=[]
classimages1=[]
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for c in range(3):
        cim=cv2.cvtColor(skimage.img_as_ubyte(skimage.transform.resize(
                skimage.io.imread("utility/pics/c{}.png".format(c)),(100,100)))[:,:,0:3],
                cv2.COLOR_RGB2BGR)
        classimages1.append(cim)
        classimages0.append(255-(255-cim)//3)

#import model
tf.keras.backend.set_learning_phase(0)
model = tf.keras.models.load_model('utility/models/sp21_cse217_v2_augmented.model')
process = tf.keras.backend.function([model.layers[0].input], [model.layers[-1].output])

#sizes image to match input size
def preprocess(frame):
    input_tensor_size = model.input_shape[1:3]
    return skimage.transform.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),input_tensor_size)

#generating probabilities of each class
def analyze(frame):
    data = process([frame[np.newaxis,:,:,:]])
    return data[0][0,:],[d[0,:] for d in data[1:]]

#dray probability distribution
def drawbars(frame, values):
    barh=300
    barw=100
    h, _ =frame.shape[0:2]
    for i,v in enumerate(values):
        x0, x1 = i*barw, i*barw+barw
        y0, y1 = h-barw, h
        cv2.rectangle(frame, (x0,h-barw), (x1, h-int(v*barh)-barw), (100,100,100), -1)
        if(v==max(values)):
            frame[y0:y1,x0:x1,:] = classimages1[i][:,:,0:3]
        else:
            frame[y0:y1,x0:x1,:] = classimages0[i][:,:,0:3]

#grabs each frame from webcam
def gen_frames(cam): 
    while True:
        success, frame = cam.read()  # read the camera frame
        if not success:
            break
        else:
            #process image
            image = frame.copy()
            h, w =  h, w = image.shape[0:2]
            im = preprocess(image[h//2-150:h//2+150, w//2-150:w//2+150,:])

            #analyze image
            out, _ = analyze(im) 

            #show results
            drawbars(image,out)

            #sending image to browser
            _ , frame = cv2.imencode('.jpg', image)
            frame = frame.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cam.release()
    cv2.destroyAllWindows()