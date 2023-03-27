# seniorIS_code

## Setting up the working directory
- After cloning, the working directory tree is:
<pre>
    .
    |-- input/
    |   `-- nuImages
    |-- output/
    |-- metadata_backup/
    |   `-- metadata-16444/
    |       |-- mask_r-cnn_lic_cmatrix.json
    |       |-- truth_class_counter.json
    |       `-- yolov5_lic_cmatrix.json
    |-- src/
    |   |-- label_mapping.py
    |   |-- main.py
    |   |-- main_utils.py
    |   |-- metrics.py
    |   |-- mrcnn_utils.py
    |   |-- nuim_util.py
    |   |-- predict_object.py
    |   |-- truth_object.py
    |   `-- yolo_utils.py
    |-- README.md
    `-- requirements.txt
</pre>

- Download the nuImage dataset at `https://www.nuscenes.org/nuimages#download`. 
For this software, we only need to download the `Metadata` and `Samples` directories 
under `All` options. We put the `Samples` directory in the `input/nuImages` directory, 
while extracting the content of `Metadata` directory to the `input/nuImages` directory. 
Once done, our directory tree is:
<pre>
    .
    |-- input/
    |   `-- nuImages/
    |       |-- samples
    |       |-- v1.0-mini
    |       |-- v1.0-test
    |       |-- v1.0-train
    |       `-- v1.0-val
    |-- output/
    |-- metadata_backup/
    |   `-- metadata-16444/
    |       |-- mask_r-cnn_lic_cmatrix.json
    |       |-- truth_class_counter.json
    |       `-- yolov5_lic_cmatrix.json
    |-- src/
    |   |-- label_mapping.py
    |   |-- main.py
    |   |-- main_utils.py
    |   |-- metrics.py
    |   |-- mrcnn_utils.py
    |   |-- nuim_util.py
    |   |-- predict_object.py
    |   |-- truth_object.py
    |   `-- yolo_utils.py
    |-- README.md
    `-- requirements.txt
</pre>

## Prepare the Python runtime environment
- Now, we can create a python virtual environment. We test our software using 
the Python3.9 but any Python version above 3.6 (need typehint and f-string)
should work just fine. A list of required package is provided in `requirements.txt`.

## Run the software
- Change the directory to `src/`. With the configured Python environment from 
the previous step, we can run the software by calling `python main.py` in the 
terminal
  - By default, the software will evaluate the Mask R-CNN and YOLOv5 models on the 
  nuImage validation set (16445 images). The output of the software is stored in 
  the `output` directory. 
    - Note on file name: 
      - \<model>_lic_cmatrix: is a json that stored confusion matrix in the following format:
      <pre>
      {
        class_label: {
          IoU_threshold: {
            confidence_threshold: {
              confusion_matrix -- 2D list
            }
          }
        }
      }        
      </pre>
      - \<model>_lic_cmatrix: is a json that stored the computed metrics in the following format:
      <pre>
      {
        class_label: {
          IoU_threshold: {
            confidence_threshold: {
              "accuracy": f
              "precision": f
              "recall": f
              "f1": f
            },
            "highest_f1": f
            "highest_f1_at_conf": f
            "AP11": f
            "AP101": f
          }
        }
      }        
      </pre>



  