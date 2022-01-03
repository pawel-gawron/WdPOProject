import cv2
import json
import click
import numpy as np

from glob import glob
from tqdm import tqdm

from typing import Dict


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
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    imgHSV = cv2.resize(imgHSV, (500, 500))
    imgHSV_blur = cv2.GaussianBlur(imgHSV, (5, 5), 0)
    cv2.imshow("imgHSV_blur", imgHSV_blur)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # by the brightness
    lowerBanana = np.array([48, 0.6*256, 256], dtype=np.int32)
    upperBanana = np.array([48, 256, 256])
    lowerApple = np.array([6, 0.57*256, 256], dtype=np.int32)
    upperApple = np.array([6, 256, 256])
    lowerOrange = np.array([30, 0.7*256, 256], dtype=np.int32)
    upperOrange = np.array([30, 256, 256])

    # by the color
    lowerBanana_2 = np.array([44, 256, 256])
    upperBanana_2 = np.array([76, 256, 256])
    lowerApple_2 = np.array([0, 256, 256])
    upperApple_2 = np.array([24, 256, 256])
    lowerOrange_2 = np.array([24, 256, 256])
    upperOrange_2 = np.array([44, 256, 256])

    bananaMask_1 = cv2.inRange(imgHSV_blur, lowerBanana, upperBanana)
    bananaMask_2 = cv2.inRange(imgHSV_blur, lowerBanana_2, upperBanana_2)
    bananaMask = bananaMask_1 + bananaMask_2

    appleMask_1 = cv2.inRange(imgHSV_blur, lowerApple, upperApple)
    appleMask_2 = cv2.inRange(imgHSV_blur, lowerApple_2, upperApple_2)
    appleMask = appleMask_1 + appleMask_2

    orangeMask_1 = cv2.inRange(imgHSV_blur, lowerOrange, upperOrange)
    orangeMask_2 = cv2.inRange(imgHSV_blur, lowerOrange_2, upperOrange_2)
    orangeMask = orangeMask_1 + orangeMask_2

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(imgHSV_blur, imgHSV_blur, mask=bananaMask)
    cv2.imshow("bananaMask", bananaMask)
    cv2.imshow("res", res)
    
    apple = 0
    banana = 0
    orange = 0



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