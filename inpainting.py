import numpy as np
from define_model import *
import sys

def define_inpainting_model(height, width):
    num_u = [128, 128, 128, 128, 128]
    num_d = [128, 128, 128, 128, 128]
    kernel_u = [3, 3, 3, 3, 3]
    kernel_d = [3, 3, 3, 3, 3]
    num_s = [0, 0, 0, 0, 0]
    kernel_s = [0, 0, 0, 0, 0]
    lr = 0.01
    inter = "bilinear"

    base_model = define_model(num_u, num_d, kernel_u, kernel_d, num_s, kernel_s, height, width, inter, lr, input_channel=2)

    masked_image = Input(shape=(height, width, 3))
    mask_image = Input(shape=(height, width, 1))

    loss = Lambda(lambda x: K.sum(K.square((x[0] - x[1]) * x[2]), axis=-1))([base_model.output, masked_image, mask_image])

    model = Model([base_model.input, masked_image, mask_image], loss)
    model.compile(loss="mean_absolute_error", optimizer=Adam(lr=lr))

    return model, base_model


def inpainting(masked_image, mask_image):
    height, width = masked_image.shape[:2]
    model, base_model = define_inpainting_model(height, width)

    y = np.linspace(0.0, 1.0, height)
    x = np.linspace(0.0, 1.0, width)
    XX, YY = np.meshgrid(x, y)
    input_meshgrid = np.concatenate([XX[:,:,None], YY[:,:,None]], axis=2)

    for i in range(5000):
        model.train_on_batch([input_meshgrid[None, :, :, :], masked_image[None, :, :, :], mask_image[None, :, :, None]], np.zeros((1, 320, 320)))

    inpainted_image = np.clip(base_model.predict(input_meshgrid[None, :, :, :])[0], 0, 255).astype(np.uint8)
    return (masked_image * mask_image[:, :, None] + inpainted_image * (1 - mask_image[:, :, None])).astype(np.uint8)


if __name__ == "__main__":
    masked_name = sys.argv[1]
    mask_name = sys.argv[2]
    inpainted_name = sys.argv[3]

    masked_image = cv2.imread(masked_name)
    mask_image = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE) / 255.0
    inpainted_image = inpainting(masked_image, mask_image)
    cv2.imwrite(inpainted_name, inpainted_image)
