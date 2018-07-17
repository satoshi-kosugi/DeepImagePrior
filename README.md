# Super-resolution
You can run the code of super-resolution as follows.
```
python super_resolution.py <LR_image_name> <SR_image_name>
```
This is an example of the result. From left to right, LR, bicubic and the proposed method.

<img src="dataset/super_resolution/Set5/image_SRF_4/img_001_SRF_4_LR.png" width="75px"> <img src="dataset/super_resolution/Set5/image_SRF_4/img_001_SRF_4_bicubic.png" width="300px"> <img src="result/super_resolution/Set5/image_SRF_4/img_001_SRF_4_LR.png" width="300px">
<br>
<br>


# Denoising
You can run the code of denoising as follows.
```
python denoising.py <noisy_image_name> <denoised_image_name>
```
This is an example of the result. A noisy image and the denoised image.

<img src="dataset/denoising/kodim03_noi_s25.png" width="400px"> <img src="result/denoising/kodim03.png" width="400px">
<br>
<br>
  
  
# Inpainting
You can run the code of inpainting as follows.
```
python inpainting.py <masked_image_name> <mask_image_name> <inpainted_image_name> 
```
This is an example of the result. A masked image and the inpainted image.

<img src="dataset/inpainting/vase_masked.png" width="300px"> <img src="result/inpainting/vase.png" width="300px">
