import sys

import numpy as np
from PIL import Image
import tensorflow as tf

if __name__ == '__main__':
    num_classes = 4
    img_size = (224, 224)
    input_shape = (224, 224, 3)

    img = Image.open(sys.argv[2]).convert('RGB')
    img = img.resize((224, 224))
    input_img = np.array(img).reshape((1, 224, 224, 3))
    print(input_img.shape)

    base_model = tf.keras.applications.ResNet50V2(
        include_top=False, input_shape=input_shape)
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)
    model.load_weights(sys.argv[1])

    predictions = model.predict(input_img)
    print(predictions)
