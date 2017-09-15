# GIF Super-Resolution

## Download Giphy Data

**`data/dnld_giphy.py`:** Download GIF data from https://giphy.com/. Usage as below. The GIFs tagged 'face' will be downloaded into `data/raw_gifs/face/`.
```
python dnld_giphy.py [tag]
python dnld_giphy.py 'face'
```

## Run Code

**`code/gif2img.py`:** Convert GIFs into PNG images. 
```
python gif2img.py
```
Convert all GIFs tagged 'face' in `data/raw_gifs/` to PNG images and store into `data/raw_imgs/`.
