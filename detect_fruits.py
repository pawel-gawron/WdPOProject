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
    imgHSV_blur = cv2.GaussianBlur(imgHSV, (11, 11), 0)
    imgHSV_blur = cv2.GaussianBlur(imgHSV_blur, (11, 11), 0)
    imgHSV_blur = cv2.GaussianBlur(imgHSV_blur, (11, 11), 0)
    cv2.imshow("img", img)

    apple = 0
    banana = 0
    orange = 0

    # # Create trackbars for color change
    # # Hue is from 0-179 for Opencv
    cv2.createTrackbar('HMin', 'img', 0, 179, empty_callback)
    cv2.createTrackbar('SMin', 'img', 0, 255, empty_callback)
    cv2.createTrackbar('VMin', 'img', 0, 255, empty_callback)
    cv2.createTrackbar('HMax', 'img', 0, 179, empty_callback)
    cv2.createTrackbar('SMax', 'img', 0, 255, empty_callback)
    cv2.createTrackbar('VMax', 'img', 0, 255, empty_callback)

    # Set default value for Max HSV trackbars
    cv2.setTrackbarPos('HMax', 'img', 179)
    cv2.setTrackbarPos('SMax', 'img', 255)
    cv2.setTrackbarPos('VMax', 'img', 255)

    hMin = sMin = vMin = 0
    hMax = 179
    sMax = vMax = 255
    # lowerBanana = np.array([hMin, sMin, vMin], dtype=np.int32)
    # upperBanana = np.array([hMax, sMax, vMax], dtype=np.int32)
    lowerBanana = np.array([23, 71, 50], dtype=np.int32)
    upperBanana = np.array([28, 255, 255], dtype=np.int32)

    # lowerApple = np.array([hMin, sMin, vMin], dtype=np.int32)
    # upperApple = np.array([hMax, sMax, vMax], dtype=np.int32)
    lowerApple = np.array([0, 112, 0], dtype=np.int32)
    upperApple = np.array([17, 220, 230], dtype=np.int32)

    # lowerOrange = np.array([hMin, sMin, vMin], dtype=np.int32)
    # upperOrange = np.array([hMax, sMax, vMax], dtype=np.int32)
    lowerOrange = np.array([12, 175, 175], dtype=np.int32)
    upperOrange = np.array([19, 255, 255], dtype=np.int32)

    bananaMask = cv2.inRange(imgHSV_blur, lowerBanana, upperBanana)
    appleMask = cv2.inRange(imgHSV_blur, lowerApple, upperApple)
    orangeMask = cv2.inRange(imgHSV_blur, lowerOrange, upperOrange)

    kernel_banana = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    maskCloseBanana = cv2.morphologyEx(bananaMask, cv2.MORPH_CLOSE, kernel_banana)
    maskOpenBanana = cv2.morphologyEx(maskCloseBanana, cv2.MORPH_OPEN, kernel_banana)

    kernel_apple = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    maskCloseApple = cv2.morphologyEx(appleMask, cv2.MORPH_CLOSE, kernel_apple)
    maskOpenApple = cv2.morphologyEx(maskCloseApple, cv2.MORPH_OPEN, kernel_apple)

    kernel_orange = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    maskCloseOrange = cv2.morphologyEx(orangeMask, cv2.MORPH_CLOSE, kernel_orange)
    maskOpenOrange = cv2.morphologyEx(maskCloseOrange, cv2.MORPH_OPEN, kernel_orange)

    contours_banana, hierarchy_banana = cv2.findContours(image=maskOpenBanana, mode=cv2.RETR_EXTERNAL,
                                                         method=cv2.CHAIN_APPROX_NONE)
    contours_apple, hierarchy_apple = cv2.findContours(image=maskOpenApple, mode=cv2.RETR_EXTERNAL,
                                                       method=cv2.CHAIN_APPROX_NONE)
    contours_orange, hierarchy_orange = cv2.findContours(image=maskOpenOrange, mode=cv2.RETR_EXTERNAL,
                                                       method=cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(imgHSV_blur, contours_orange, -1, (255, 0, 0), 3)

    bgrMaskBanana = cv2.cvtColor(maskOpenBanana, cv2.COLOR_GRAY2BGR)
    imgFinalBanana = cv2.addWeighted(bgrMaskBanana, 0.5, img, 0.5, 0)

    bgrMaskApple = cv2.cvtColor(maskOpenApple, cv2.COLOR_GRAY2BGR)
    imgFinalApple = cv2.addWeighted(bgrMaskApple, 0.5, img, 0.5, 0)

    bgrMaskOrange = cv2.cvtColor(maskOpenOrange, cv2.COLOR_GRAY2BGR)
    imgFinalOrange = cv2.addWeighted(bgrMaskOrange, 0.5, img, 0.5, 0)

    cv2.imshow("imgFinal", imgFinalOrange)
    cv2.imshow("maskOpen",maskOpenOrange)
    cv2.imshow("imgHSV_blur", imgHSV_blur)

    while True:
        img2 = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img2 = cv2.resize(img2, (600, 600))
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        imgHSV_blur = cv2.GaussianBlur(imgHSV, (11, 11), 0)
        imgHSV_blur = cv2.GaussianBlur(imgHSV_blur, (11, 11), 0)
        imgHSV_blur = cv2.GaussianBlur(imgHSV_blur, (11, 11), 0)

        hMin = cv2.getTrackbarPos('HMin', 'img')
        sMin = cv2.getTrackbarPos('SMin', 'img')
        vMin = cv2.getTrackbarPos('VMin', 'img')
        hMax = cv2.getTrackbarPos('HMax', 'img')
        sMax = cv2.getTrackbarPos('SMax', 'img')
        vMax = cv2.getTrackbarPos('VMax', 'img')
        # by the brightness
        lowerBanana = np.array([23, 71, 50], dtype=np.int32)
        upperBanana = np.array([28, 255, 255], dtype=np.int32)
        # lowerBanana = np.array([hMin, sMin, vMin], dtype=np.int32)
        # upperBanana = np.array([hMax, sMax, vMax], dtype=np.int32)
        # lowerApple = np.array([hMin, sMin, vMin], dtype=np.int32)
        # upperApple = np.array([hMax, sMax, vMax], dtype=np.int32)
        lowerApple = np.array([0, 112, 0], dtype=np.int32)
        upperApple = np.array([17, 218, 225], dtype=np.int32)
        lowerOrange = np.array([12, 175, 175], dtype=np.int32)
        upperOrange = np.array([19, 255, 255], dtype=np.int32)
        # lowerOrange = np.array([hMin, sMin, vMin], dtype=np.int32)
        # upperOrange = np.array([hMax, sMax, vMax], dtype=np.int32)

        bananaMask = cv2.inRange(imgHSV_blur, lowerBanana, upperBanana)

        appleMask = cv2.inRange(imgHSV_blur, lowerApple, upperApple)

        orangeMask = cv2.inRange(imgHSV_blur, lowerOrange, upperOrange)

        kernel_banana = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
        maskCloseBanana = cv2.morphologyEx(bananaMask, cv2.MORPH_CLOSE, kernel_banana)
        maskOpenBanana = cv2.morphologyEx(maskCloseBanana, cv2.MORPH_OPEN, kernel_banana)

        kernel_apple = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        maskCloseApple = cv2.morphologyEx(appleMask, cv2.MORPH_CLOSE, kernel_apple)
        maskOpenApple = cv2.morphologyEx(maskCloseApple, cv2.MORPH_OPEN, kernel_apple)

        kernel_orange = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        maskCloseOrange = cv2.morphologyEx(orangeMask, cv2.MORPH_CLOSE, kernel_orange)
        maskOpenOrange = cv2.morphologyEx(maskCloseOrange, cv2.MORPH_OPEN, kernel_orange)

        contours_banana, hierarchy_banana = cv2.findContours(image=maskOpenBanana, mode=cv2.RETR_EXTERNAL,
                                                             method=cv2.CHAIN_APPROX_NONE)
        contours_apple, hierarchy_apple = cv2.findContours(image=maskOpenApple, mode=cv2.RETR_EXTERNAL,
                                                             method=cv2.CHAIN_APPROX_NONE)
        contours_orange, hierarchy_orange = cv2.findContours(image=maskOpenOrange, mode=cv2.RETR_EXTERNAL,
                                                             method=cv2.CHAIN_APPROX_NONE)
        # print("kontury: ", contours)
        # print("rozmiar: ", len(contours))
        # print(banana)
        for i in range(len(contours_banana)):
            if len(contours_banana[i]) > 100:
                banana = banana + 1
                # print("wektor: ", len(contours_banana[i]))

        for i in range(len(contours_apple)):
            if len(contours_apple[i]) > 100:
                apple = apple + 1
                print("wektor: ", len(contours_apple[i]))

        for i in range(len(contours_orange)):
            if len(contours_orange[i]) > 200:
                orange = orange + 1
                print("wektor: ", len(contours_orange[i]))

        cv2.drawContours(imgHSV_blur, contours_orange, -1, (255, 0, 0), 3)

        bgrMaskBanana = cv2.cvtColor(maskOpenBanana, cv2.COLOR_GRAY2BGR)
        imgFinalBanana = cv2.addWeighted(bgrMaskBanana, 0.5, img, 0.5, 0)

        bgrMaskApple = cv2.cvtColor(maskOpenApple, cv2.COLOR_GRAY2BGR)
        imgFinalApple = cv2.addWeighted(bgrMaskApple, 0.5, img, 0.5, 0)

        bgrMaskOrange = cv2.cvtColor(maskOpenOrange, cv2.COLOR_GRAY2BGR)
        imgFinalOrange = cv2.addWeighted(bgrMaskOrange, 0.5, img, 0.5, 0)

        cv2.imshow("imgFinal", imgFinalOrange)
        cv2.imshow("maskOpen", maskOpenOrange)
        cv2.imshow("img", img2)
        cv2.imshow("imgHSV_blur", imgHSV_blur)
        # cv2.imshow("imgOwoce", img2[contours_orange])

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cv2.waitKey(0)
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