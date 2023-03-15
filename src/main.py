import main_utils as mut

import nuim_util as nuu

IMG_INDEX = range(0, 1)


def main():
    nuim_obj, mrcnn_model, yolo_model = mut.initialization()

    for img_index in IMG_INDEX:
        img_path, truth_objs = nuu.get_truth_objs(nuim_obj, img_index)

        mrcnn_preds = mut.get_mrcnn_predict_objs(mrcnn_model, img_path)
        print("-" * 50, len(mrcnn_preds))
        for obj in mrcnn_preds:
            print(obj)

        yolo_preds = mut.get_yolo_predict_objs(yolo_model, img_path)
        print("-" * 50, len(yolo_preds))
        for obj in yolo_preds:
            print(obj)


main()
