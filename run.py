import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

def main():

    img_path = str(input('Введите путь к изображению: '))
    model_path = str(input('Введите путь к модели tensorflow: '))
    
    model = load_model(model_path)

    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.resnet50.preprocess_input(x)
    x /= 255.

    preds = model.predict(x)

    print(preds[0])

if __name__ == '__main__':
    main()