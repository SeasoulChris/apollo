# Google Image Downloader

1. Setup

   ```bash
   pip install google_images_download
   ```

1. Edit and generate config

   ```bash
   ./config_generator.py
   ```

1. Download

   ```bash
   googleimagesdownload -cf google-image-downloader-conf.json
   ```
   You will find the images in `./downloads` folder.

1. Reorg images to be programming friendly.

   ```bash
   ./reorg_images.sh ./downloads ./images
   ```
