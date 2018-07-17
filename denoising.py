import numpy as np
from define_model import *
import sys

def squared_error(y_true, y_pred):
    return K.sum(K.square(y_pred - y_true))

def define_denoising_model(height, width):
    num_u = [128, 128, 128, 128, 128]
    num_d = [128, 128, 128, 128, 128]
    kernel_u = [3, 3, 3, 3, 3]
    kernel_d = [3, 3, 3, 3, 3]
    num_s = [4, 4, 4, 4, 4]
    kernel_s = [1, 1, 1, 1, 1]
    lr = 0.01
    inter = "bilinear"

    model = define_model(num_u, num_d, kernel_u, kernel_d, num_s, kernel_s, height, width, inter, lr)
    model.compile(loss=squared_error, optimizer=Adam(lr=lr))

    return model


def denoising(image):
    height, width = image.shape[:2]
    model = define_denoising_model(height, width)
    input_noise = np.random.uniform(0, 0.1, (1, height, width, 32))

    for i in range(1800):
        model.train_on_batch(input_noise + np.random.normal(0, 1/30.0, (height, width, 32)), image[None, :, :, :])

    return np.clip(model.predict(input_noise)[0], 0, 255).astype(np.uint8)


if __name__ == "__main__":
    noisy_name = sys.argv[1]
    denoised_name = sys.argv[2]

    noisy_image = cv2.imread(noisy_name)
    denoised_image = denoising(noisy_image)
    cv2.imwrite(denoised_name, denoised_image)
