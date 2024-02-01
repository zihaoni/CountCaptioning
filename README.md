# CountCaptioning


This is an experiment about CountCaptioning. 

The addition to traditional captioning is following:
- object count information
- new metrics
- initial modules for futher study


## Requirements
Python 3.6
PyTorch 1.4 (along with torchvision)
cider (already been added as a submodule)


# Training for preliminaly captioning module:

```
python train.py --id $id --caption_model fc --noamopt --noamopt_warmup 20000 --label_smoothing 0.0 --input_json data/cocotalk.json --input_label_h5 data/cocotalk_label.h5 --input_fc_dir data/cocobu_fc --input_att_dir data/cocobu_att --seq_per_img 5 --batch_size 10 --beam_size 1 --learning_rate 5e-4 --num_layers 6 --input_encoding_size 512 --rnn_size 2048 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --checkpoint_path $ckpt_path $start_from --save_checkpoint_every 3000 --language_eval 1 --val_images_use 5000 --max_epochs 15

python train.py --id $id --caption_model fc --reduce_on_plateau --input_json data/cocotalk.json --input_label_h5 data/cocotalk_label.h5 --input_fc_dir data/cocobu_fc --input_att_dir data/cocobu_att --input_box_dir data/cocobu_box --seq_per_img 5 --batch_size 10 --beam_size 1 --learning_rate 1e-5 --num_layers 6 --input_encoding_size 512 --rnn_size 2048 --checkpoint_path $ckpt_path $start_from --save_checkpoint_every 3000 --language_eval 1 --val_images_use 5000 --self_critical_after 10
```

**Notice**: Set the hyperparameters for fc:
```
N=num_layers
d_model=input_encoding_size
d_ff=rnn_size
h is always 8
```

# Training for Object counting module:


Downloading Official Pre-trained Weights:
For easy demo purposes we will use the pre-trained weights.
Download pre-trained weights file: https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT

Copy and paste yolov4.weights from your downloads folder into the 'data' folder of this repository.


Using Custom Trained YOLO Weights:

Copy and paste your custom .weights file into the 'data' folder and copy and paste your custom .names into the 'data/classes/' folder.



To implement object detection using TensorFlow, first we convert the .weights into the corresponding TensorFlow model files and then run the model.
```bash
# Convert darknet weights to tensorflow

python save_model.py --weights ./data/weights --output ./checkpoints/yolo-416 --input_size 416 --model yolo 

# Run yolo tensorflow model
python detect.py --weights ./checkpoints/yolo-416 --size 416 --model yolo --images ./data/images/kite.jpg
```




# Configuring numeral editing module and overall Evaluation

**Note**: Place all your images of interest into a folder, e.g. `blah`, and run
the eval script:

```bash
$ python tools/eval.py --model model.pth --infos_path infos.pkl --image_folder blah --num_images 10
```
