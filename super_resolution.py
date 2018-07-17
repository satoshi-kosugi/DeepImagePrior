import numpy as np
from define_model import *
import sys

def squared_error(y_true, y_pred):
    return K.sum(K.square(y_pred - y_true))

def define_super_resolution_model(height, width):
    num_u = [128, 128, 128, 128, 128]
    num_d = [128, 128, 128, 128, 128]
    kernel_u = [3, 3, 3, 3, 3]
    kernel_d = [3, 3, 3, 3, 3]
    num_s = [4, 4, 4, 4, 4]
    kernel_s = [1, 1, 1, 1, 1]
    lr = 0.01
    inter = "bilinear"

    base_model = define_model(num_u, num_d, kernel_u, kernel_d, num_s, kernel_s, height, width, inter, lr)

    lanczos_kernel = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            x_d = np.abs(j-1.5)
            y_d = np.abs(i-1.5)
            lanczos_kernel[i,j] = np.sinc(x_d) * np.sinc(x_d/2.0) + np.sinc(y_d) * np.sinc(y_d/2.0)
    lanczos_kernel = lanczos_kernel / lanczos_kernel.sum()


    x = base_model.output
    down_sampled = Lambda(lambda x: K.zeros_like(x[:, ::4, ::4, :]))(x)

    for i in range(4):
        for j in range(4):
            down_sampled = Lambda(lambda x: x[0][:, i::4, j::4, :] * lanczos_kernel[i, j] + x[1])([x, down_sampled])

    model = Model(base_model.input, down_sampled)
    model.compile(loss=squared_error, optimizer=Adam(lr=lr))

    return model, base_model


def super_resolution(image):
    height, width = image.shape[:2]
    height *= 4
    width *= 4
    model, base_model = define_super_resolution_model(height, width)
    input_noise = np.random.uniform(0, 0.1, (1, height, width, 32))

    for i in range(2000):
        model.train_on_batch(input_noise + np.random.normal(0, 1/30.0, (height, width, 32)), image[None, :, :, :])

    return np.clip(base_model.predict(input_noise)[0], 0, 255).astype(np.uint8)


if __name__ == "__main__":
    lr_name = sys.argv[1]
    sr_name = sys.argv[2]

    lr_image = cv2.imread(lr_name)
    sr_image = super_resolution(lr_image)
    cv2.imwrite(sr_name, sr_image)
