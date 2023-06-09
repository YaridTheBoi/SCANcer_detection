WORKSPACE_PATH = 'Tensorflow/workspace'
SCRIPTS_PATH = 'Tensorflow/scripts'
APIMODEL_PATH = 'Tensorflow/models'
ANNOTATIONS_PATH = WORKSPACE_PATH + '/annotations'
IMAGE_PATH = WORKSPACE_PATH + '/images'                         
MODEL_PATH = WORKSPACE_PATH + '/models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH + '/pre-trained-models'
CONFIG_PATH = MODEL_PATH + '/my_ssd_mobnet/pipeline.config' 
CHECKPOINT_PATH = MODEL_PATH + '/my_ssd_mobnet/'
CUSTOM_MODEL_NAME = 'my_ssd_mobnet'

import cv2
import numpy as np
import random
import sys
import os
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_utils
from object_detection.builders import model_builder


import tensorflow as tf
from object_detection.utils import config_util




#zaladowanie modelu i zbuildowanie go wg configu
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs["model"], is_training=False)

#zaladowanie ostatniego checkpointa (najnowszy stan wiedzy modelu)
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH , 'ckpt-6')).expect_partial()


# funkcja tensorflowa do wykrywania obiektow na zdjeciu
@tf.function
def detect_fn(image):
    image,shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


def analyze(filename):
    # plik z etykietami 
    category_index = label_map_util.create_category_index_from_labelmap(ANNOTATIONS_PATH + '/label_map.pbtxt')


    img = cv2.imread(cv2.samples.findFile(filename))
    img_height = img.shape[0]
    img_width = img.shape[1]
    if img is None:
        print("Could not read the image.")
        return



    #detekcja
    input_tensor = tf.convert_to_tensor(np.expand_dims(img, 0), dtype =tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))

    detections = {key:value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    label_id_offset =1



    # informacje z wykrycia
    print("\nIMAGE SIZE:")
    print("HEIGHT: "+ str(img_height))
    print("WIDTH: "+ str(img_width))

    print('\nBOXES:')
    print(detections['detection_boxes'][0])

    print('\nBOXES COORDINATES:')
    y1 = int(detections['detection_boxes'][0][0] * img_height)
    x1 = int(detections['detection_boxes'][0][1] * img_width)

    y2 = int(detections['detection_boxes'][0][2] * img_height)
    x2 = int(detections['detection_boxes'][0][3] * img_width)

    print("x1: {}, y1:{}".format(x1,y1))
    print("x2: {}, y2:{}".format(x2,y2))

    print('\nSCORES:')
    print(detections['detection_scores'])


    cut_img = img[ y1:y2+10, x1-10:x2+10]
    # wykryte zmiany odseparowane
    cv2.imshow("Separated Changes", cut_img)
    
    cut_img_grayscale = cv2.cvtColor(cut_img, cv2.COLOR_BGR2GRAY)

    #obraz trzeba tak zmiekczac az znajde jeden kontur
    
    contours_amount = 100

    while contours_amount!=1:
        cut_img_grayscale = cv2.GaussianBlur(cut_img_grayscale, (5,5), 0) 
        _, cut_img_binary = cv2.threshold(cut_img_grayscale,150,255, 0)

        cut_img_edges = cv2.Canny(cut_img_binary, 125, 255)
        contours, _ = cv2.findContours(cut_img_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_amount = len(contours)


    cv2.imshow("Binary", cut_img_binary)

    cv2.imshow("Edges", cut_img_edges)

    img_contoures = cut_img.copy()
    img_contoures = cv2.drawContours(img_contoures, contours, -1, (255,0,255), 2)
    cv2.imshow("Contours", img_contoures)

    # aproksymacja konturu i obliczanie końcowych parametrów
    perimeter = cv2.arcLength(contours[0], True)
    epsilon = 0.1 * perimeter
    approx = cv2.approxPolyDP(contours[0], epsilon, True)

    # obliczenie współczynników koła dla porównania z kształtem plamy
    center, radius = cv2.minEnclosingCircle(contours[0])
    circularity = 4 * np.pi * cv2.contourArea(contours[0]) / (perimeter * perimeter)

    print('\nCIRCLE SIMILARITY%')
    print(circularity)


    M = cv2.moments(contours[0])
    hu_moments = cv2.HuMoments(M)
    print("\nSymmetry:", hu_moments[2])





    # zaznaczenie znalezionych obszarow
    vis_utils.visualize_boxes_and_labels_on_image_array(
        img,
        detections['detection_boxes'],
        detections['detection_classes']+label_id_offset,
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw= 1,
        min_score_thresh=0.15,
        agnostic_mode=False
    )


    #wykryte zmiany
    cv2.imshow("Detection", img)
    


    k = cv2.waitKey(0)
    if k == ord("s"):
        cv2.imwrite("analyzed_"+filename, img)
    elif k == ord("q"):
        return


if __name__ == "__main__":
    analyze(*sys.argv[1:])