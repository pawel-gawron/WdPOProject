import cv2
import json
import click
import numpy as np

from glob import glob
from tqdm import tqdm

from typing import Dict

def empty_callback(value):
    print(f'Trackbar reporting for duty with value: {value}')
    # pass

def detect_fruits(img_path: str) -> Dict[str, int]:
    """Fruit detection function, to implement.

    Parameters
    ----------
    img_path : str
        Path to processed image.

    Returns
    -------
    Dict[str, int]
        Dictionary with quantity of each fruit.
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    #TODO: Implement detection method.
    img = cv2.resize(img, (600, 600))
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    imgHSV_blur = cv2.GaussianBlur(imgHSV, (7, 7), 0)
    cv2.imshow("imgHSV_blur", imgHSV_blur)

    apple = 0
    banana = 0
    orange = 0

    # Create trackbars for color change
    # Hue is from 0-179 for Opencv
    cv2.createTrackbar('HMin', 'imgHSV_blur', 0, 179, empty_callback)
    cv2.createTrackbar('SMin', 'imgHSV_blur', 0, 255, empty_callback)
    cv2.createTrackbar('VMin', 'imgHSV_blur', 0, 255, empty_callback)
    cv2.createTrackbar('HMax', 'imgHSV_blur', 0, 179, empty_callback)
    cv2.createTrackbar('SMax', 'imgHSV_blur', 0, 255, empty_callback)
    cv2.createTrackbar('VMax', 'imgHSV_blur', 0, 255, empty_callback)

    # Set default value for Max HSV trackbars
    cv2.setTrackbarPos('HMax', 'imgHSV_blur', 179)
    cv2.setTrackbarPos('SMax', 'imgHSV_blur', 255)
    cv2.setTrackbarPos('VMax', 'imgHSV_blur', 255)

    hMin = sMin = vMin = 0
    hMax = 179
    sMax = vMax = 255
    lowerBanana = np.array([hMin, sMin, vMin], dtype=np.int32)
    upperBanana = np.array([hMax, sMax, vMax], dtype=np.int32)

    bananaMask_1 = cv2.inRange(imgHSV_blur, lowerBanana, upperBanana)
    bananaMask = bananaMask_1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    maskCloseBanana = cv2.morphologyEx(bananaMask, cv2.MORPH_CLOSE, kernel)
    maskOpenBanana = cv2.morphologyEx(maskCloseBanana, cv2.MORPH_OPEN, kernel)

    contours, hierarchy = cv2.findContours(image=maskOpenBanana, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(imgHSV_blur, contours, -1, (255, 255, 255), 5)

    bgrMaskBanana = cv2.cvtColor(maskOpenBanana, cv2.COLOR_GRAY2BGR)
    imgFinal = cv2.addWeighted(bgrMaskBanana, 0.5, img, 0.5, 0)

    cv2.imshow("imgFinal", imgFinal)
    cv2.imshow("maskOpenBanana", maskOpenBanana)

    while True:
        cv2.imshow("imgHSV_blur", imgHSV_blur)
        hMin = cv2.getTrackbarPos('HMin', 'imgHSV_blur')
        sMin = cv2.getTrackbarPos('SMin', 'imgHSV_blur')
        vMin = cv2.getTrackbarPos('VMin', 'imgHSV_blur')
        hMax = cv2.getTrackbarPos('HMax', 'imgHSV_blur')
        sMax = cv2.getTrackbarPos('SMax', 'imgHSV_blur')
        vMax = cv2.getTrackbarPos('VMax', 'imgHSV_blur')
        # by the brightness
        # lowerBanana = np.array([48/2, 0.6*255, 255], dtype=np.int32)
        # upperBanana = np.array([48/2, 255, 255], dtype=np.int32)
        lowerBanana = np.array([hMin, sMin, vMin], dtype=np.int32)
        upperBanana = np.array([hMax, sMax, vMax], dtype=np.int32)
        lowerApple = np.array([6 / 2, 0.57 * 255, 255], dtype=np.int32)
        upperApple = np.array([6 / 2, 255, 255], dtype=np.int32)
        lowerOrange = np.array([30 / 2, 0.7 * 255, 255], dtype=np.int32)
        upperOrange = np.array([30 / 2, 255, 255], dtype=np.int32)

        # by the color
        # lowerBanana_2 = np.array([44/2, 255, 255], dtype=np.int32)
        # upperBanana_2 = np.array([76/2, 255, 255], dtype=np.int32)
        lowerBanana_2 = np.array([hMin, 255, 255], dtype=np.int32)
        upperBanana_2 = np.array([hMax, 255, 255], dtype=np.int32)
        lowerApple_2 = np.array([0, 255, 255], dtype=np.int32)
        upperApple_2 = np.array([24/2, 255, 255], dtype=np.int32)
        lowerOrange_2 = np.array([24/2, 255, 255], dtype=np.int32)
        upperOrange_2 = np.array([44/2, 255, 255], dtype=np.int32)

        bananaMask_1 = cv2.inRange(imgHSV_blur, lowerBanana, upperBanana)
        bananaMask_2 = cv2.inRange(imgHSV_blur, lowerBanana_2, upperBanana_2)
        bananaMask = bananaMask_1

        appleMask_1 = cv2.inRange(imgHSV_blur, lowerApple, upperApple)
        appleMask_2 = cv2.inRange(imgHSV_blur, lowerApple_2, upperApple_2)
        appleMask = appleMask_1 + appleMask_2

        orangeMask_1 = cv2.inRange(imgHSV_blur, lowerOrange, upperOrange)
        orangeMask_2 = cv2.inRange(imgHSV_blur, lowerOrange_2, upperOrange_2)
        orangeMask = orangeMask_1 + orangeMask_2

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        maskCloseBanana = cv2.morphologyEx(bananaMask, cv2.MORPH_CLOSE, kernel)
        maskOpenBanana = cv2.morphologyEx(maskCloseBanana, cv2.MORPH_OPEN, kernel)

        contours, hierarchy = cv2.findContours(image=maskOpenBanana, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

        cv2.drawContours(imgHSV_blur, contours, -1, (255, 255, 255), 5)

        bgrMaskBanana = cv2.cvtColor(maskOpenBanana, cv2.COLOR_GRAY2BGR)
        imgFinal = cv2.addWeighted(bgrMaskBanana, 0.5, img, 0.5, 0)


        cv2.imshow("imgFinal", imgFinal)
        cv2.imshow("maskOpenBanana", maskOpenBanana)

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


    return {'apple': apple, 'banana': banana, 'orange': orange}


@click.command()
@click.option('-p', '--data_path', help='Path to data directory')
@click.option('-o', '--output_file_path', help='Path to output file')
def main(data_path, output_file_path):

    img_list = glob(f'{data_path}/*.jpg')

    results = {}

    for img_path in tqdm(sorted(img_list)):
        fruits = detect_fruits(img_path)

        filename = img_path.split('/')[-1]

        results[filename] = fruits
    
    with open(output_file_path, 'w') as ofp:
        json.dump(results, ofp)


if __name__ == '__main__':
    main()