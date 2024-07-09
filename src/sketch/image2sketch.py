
import cv2
import os
import numpy as np

def center_crop_image(image, crop_size):
    h, w, _ = image.shape
    crop_h, crop_w = crop_size

    # 中央の位置を計算
    start_x = w//2 - crop_w//2
    start_y = h//2 - crop_h//2

    # クロップ領域の端を計算
    end_x = start_x + crop_w
    end_y = start_y + crop_h

    # 画像をクロップ
    cropped_image = image[start_y:end_y, start_x:end_x]
    return cropped_image

def centerCrop_images(image_path_list, center_crop_size=(224, 224)):
    for image_path in image_path_list:
        # 画像を読み込む
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to load image: {image_path}")
            continue
        
        
        cropped_img = center_crop_image(img, center_crop_size)
        resized_img = cv2.resize(cropped_img, (224, 224))

        cv2.imwrite(image_path, resized_img)

    return 

def image2sketch(image_path_list):
    for image_path in image_path_list:
        # 画像を読み込む
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to load image: {image_path}")
            continue

        d = 15
        sigmaColor = 80
        sigmaSpace = 80
        # バイラテラルフィルタの適用
        image = cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        test = cv2.bilateralFilter(edges, d, sigmaColor, sigmaSpace)
        test = cv2.bitwise_not(test)
        rgb_edges = cv2.cvtColor(test, cv2.COLOR_GRAY2RGB)

        #image_pathのファイル名を変える
        # ディレクトリ名とファイル名に分割
        dir_name, file_name = os.path.split(image_path)
        name, ext = os.path.splitext(file_name)
        new_file_name = f"{name}_sketch{ext}"
        new_file_path = os.path.join(dir_name, new_file_name)

        cv2.imwrite(new_file_path, rgb_edges)

    return
