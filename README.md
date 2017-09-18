# GIF Super-Resolution

## Download Giphy Data

**`data/dnld_giphy.py`:** Download GIF data from https://giphy.com/. Usage as below. The GIFs tagged 'face' will be downloaded into `data/raw_gifs/face/`.
```
python dnld_giphy.py [tag]
python dnld_giphy.py 'face'
```

## Run Code

**`code/gif2img.py`:** Convert GIFs into PNG images. Convert all GIFs tagged 'face' in `data/raw_gifs/` to PNG images and store into `data/raw_imgs/`.
```
python gif2img.py
```

**`code/gen_hr_lr.py`:** Generate high-resolution (GT) and low-resolution (test input) images. HR images are stored in `data/hr_imgs/` and LR images are stored in `data/lr_imgs/`.
```
python gen_hr_lr.py
```

**`code/bicu_inter.py`:** Generate Bicubic Interpolation of low-resolution images. BI images are stored in `data/bi_imgs/`.
```
python bicu_inter.py
```

**`code/pr_sr.py`:** Implement PixelCNNs to do frame super-resolution.

**`code/tem_model.py`:** Run temporal regularization network.

### Train PixelCNN++ Model

Generate image list.
```
python prsr/tools/create_img_lists.py --dataset=../data/hr_imgs --trainfile=../data/train.txt --testfile=../data/test.txt
```
Train model.
```
python prsr/tools/train.py --device_id=0
```
Run test images.
```
python prsr/tools/test.py --device_id=0
```
