import numpy as np
import json
import onnxruntime
import tensorflow as tf
from tensorflow.keras.preprocessing import image


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def process_image(image_path, height=224, width=224):
    img = image.load_img(image_path, target_size=(width, height))
    np_image = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    np_image = tf.keras.applications.resnet50.preprocess_input(x)
    np_image /= 255.

    return [np_image]


def get_session(model_path):
    global ONNX_SESSION
    if ONNX_SESSION == None:
        sess = onnxruntime.InferenceSession(model_path)
        ONNX_SESSION = sess
    return ONNX_SESSION


def predict(img_path, model_path, use_array=False, *args, **kwargs):
    onnx_sess = get_session(model_path)
    sess_inputs = onnx_sess.get_inputs()[0]
    input_name = sess_inputs.name
    shape = sess_inputs.shape
    im = process_image(img_path, height=shape[1], width=shape[2]) 
    inference_preds = onnx_sess.run(None, {input_name: im}) # this is where the inference_happens
    results = inference_preds[0][0]
    data = {str(k):v for k,v in enumerate(results)}
    return json.dumps(data, cls=NumpyEncoder)


def main():

    img_path = str(input('Введите путь к изображению: '))
    if img_path == '':
        img_path = r'\\lsgroup.local\ekb\Public\classificator_inference_test\1ABOM_10000105_0_0_as_-1.0_53.11_498.0_340.0_28_52_.png'
        print(img_path)

    model_path = str(input('Введите путь к модели onnx: '))
    if model_path == '':
        model_path = r'\\lsgroup.local\ekb\Public\classificator_inference_test\model_as_pipelab_prep.onnx'
        print(model_path)

    preds = predict(img_path, model_path)
    print(preds)


ONNX_SESSION = None


if __name__ == "__main__":
    main()