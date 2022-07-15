# EfficientDet-DE-RoadSigns-108classes
Training and detection RoadSigns in Germany by EfficientDet

<h2>
EfficientDet DE RoadSigns 108classes (Updated: 2022/07/15)
</h2>

This is a simple python example to train and detect RoadSigns in Germany based on 
<a href="https://github.com/google/automl/tree/master/efficientdet">Google Brain AutoML efficientdet</a>.
<br>
<h2>
1. Installing tensorflow on Windows11
</h2>
We use Python 3.8.10 to run tensoflow 2.8.0 on Windows11.<br>
<h3>1.1 Install Microsoft Visual Studio Community</h3>
Please install <a href="https://visualstudio.microsoft.com/ja/vs/community/">Microsoft Visual Studio Community</a>, 
which can be used to compile source code of 
<a href="https://github.com/cocodataset/cocoapi">cocoapi</a> for PythonAPI.<br>
<h3>1.2 Create a python virtualenv </h3>
Please run the following command to create a python virtualenv of name <b>py38-efficientdet</b>.
<pre>
>cd c:\
>python38\python.exe -m venv py38-efficientdet
>cd c:\py38-efficientdet
>./scripts/activate
</pre>
<h3>1.3 Create a working folder </h3>
Please create a working folder "c:\google" for your repository, and install the python packages.<br>

<pre>
>mkdir c:\google
>cd    c:\google
>pip install cython
>git clone https://github.com/cocodataset/cocoapi
>cd cocoapi/PythonAPI
</pre>
You have to modify extra_compiler_args in setup.py in the following way:
<pre>
   extra_compile_args=[]
</pre>
<pre>
>python setup.py build_ext install
</pre>

<br>

<br>
<h2>
2. Installing EfficientDet-DE-RoadSigns
</h2>
<h3>2.1 clone EfficientDet-Slightly-Realistic-DE-RoadSigns-108classes</h3>

Please clone EfficientDet-DE-RoadSigns-108classes in the working folder <b>c:\google</b>.<br>
<pre>
>git clone  https://github.com/atlan-antillia/EfficientDet-Slightly-Realistic-DE-RoadSigns-108classes.git<br>
</pre>
You can see the following folder <b>projects</b> in  EfficientDet-Slightly-Realistic-DE-RoadSigns-108classes folder of the working folder.<br>

<pre>
EfficientDet-Slightly-Realistic-DE-RoadSigns-108classes
└─projects
    └─DE_RoadSigns
        ├─eval
        ├─saved_model
        │  └─variables
        ├─realistic-test-dataset
        └─realistic-test-dataset_outputs        
</pre>

<br>
<h3>2.2 Install python packages</h3>

Please run the following command to install python packages for this project.<br>
<pre>
>cd ./EfficientDet-Slightly-Realistic-DE-RoadSigns-108classes
>pip install -r requirements.txt
</pre>

<h3>2.3 Download TFRecord dataset</h3>
 You can download TFRecord_DE_RoadSigns 108classes dataset from 
<a href="https://drive.google.com/file/d/1OC8b0fmc7cUe8JzHW3TVRdG3hD50F-aX/view?usp=sharing">TFRecord_DE_RoadSigns_108classes_V7.1</a>
<br>
The downloaded train and valid dataset must be placed in ./projects/DE_RoadSigns folder.
<pre>
└─projects
    └─DE_RoadSigns
        ├─train
        └─valid
</pre>
<br>


<h3>2.4 Workarounds for Windows</h3>
As you know or may not know, the efficientdet scripts of training a model and creating a saved_model do not 
run well on Windows environment in case of tensorflow 2.8.0(probably after the version 2.5.0) as shown below:. 
<pre>
INFO:tensorflow:Saving checkpoints for 0 into ./models\model.ckpt.
I0609 06:22:50.961521  3404 basic_session_run_hooks.py:634] Saving checkpoints for 0 into ./models\model.ckpt.
2022-06-09 06:22:52.780440: W tensorflow/core/framework/op_kernel.cc:1745] OP_REQUIRES failed at save_restore_v2_ops.cc:110 :
 NOT_FOUND: Failed to create a NewWriteableFile: ./models\model.ckpt-0_temp\part-00000-of-00001.data-00000-of-00001.tempstate8184773265919876648 :
</pre>

The real problem seems to happen in the original <b> save_restore_v2_ops.cc</b>. The simple workarounds to the issues are 
to modify the following tensorflow/python scripts in your virutalenv folder. 
<pre>
c:\py38-efficientdet\Lib\site-packages\tensorflow\python\training
 +- basic_session_run_hooks.py
 
634    logging.info("Saving checkpoints for %d into %s.", step, self._save_path)
635    ### workaround date="2022/06/18" os="Windows"
636    import platform
637    if platform.system() == "Windows":
638      self._save_path = self._save_path.replace("/", "\\")
639    #### workaround
</pre>

<pre>
c:\py38-efficientdet\Lib\site-packages\tensorflow\python\saved_model
 +- builder_impl.py

595    variables_path = saved_model_utils.get_variables_path(self._export_dir)
596    ### workaround date="2022/06/18" os="Windows" 
597    import platform
598    if platform.system() == "Windows":
599      variables_path = variables_path.replace("/", "\\")
600    ### workaround
</pre>

<br>

<h3>3. Inspect tfrecord</h3>
 Move to ./projects/DE_RoadSigns directory, 
 and run the following bat file to inspect train/train.tfrecord:
<pre>
tfrecord_inspect.bat
</pre>
, which is the following:
<pre>
python ../../TFRecordInspector.py ^
  ./train/*.tfrecord ^
  ./label_map.pbtxt ^
  ./Inspector/train
</pre>
<br>
This will generate annotated images with bboxes and labels from the tfrecord, and cout the number of annotated objects in it.<br>
<br>
<b>TFRecordInspecotr: annotated images in train.tfrecord</b><br>
<img src="./asset/tfrecord_inspector_annotated_images.png">

<br>
<br>
<b>TFRecordInspecotr: objects_count train.tfrecord</b><br>
<img src="./asset/tfrecord_inspector_objects_count.png">
<br>
This bar graph shows that the number of the objects.
<br>
<br>
<br>
<h3>4. Downloading the pretrained-model efficientdet-d0</h3>
Please download an EfficientDet model chekcpoint file <b>efficientdet-d0.tar.gz</b>, and expand it in 
<b>EfficientDet-Slightly-Realistic-DE-RoadSigns-108classes</b> folder.<br>
<br>
https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d0.tar.gz
<br>
See: https://github.com/google/automl/tree/master/efficientdet<br>


<h3>5. Training DE RoadSigns Model by using pretrained-model</h3>
Move to the ./projects/DE_RoadSigns directory, and run the following bat file to train roadsigns efficientdet model:
<pre>
1_train.bat
</pre> 
, which is the following:
<pre>
rem 1_train.bat
python ../../ModelTrainer.py ^
  --mode=train_and_eval ^
  --train_file_pattern=./train/*.tfrecord  ^
  --val_file_pattern=./valid/*.tfrecord ^
  --model_name=efficientdet-d0 ^
  --hparams="input_rand_hflip=False,image_size=512x512,num_classes=108,label_map=./label_map.yaml" ^
  --model_dir=./models ^
  --label_map_pbtxt=./label_map.pbtxt ^
  --eval_dir=./eval ^
  --ckpt=../../efficientdet-d0  ^
  --train_batch_size=4 ^
  --early_stopping=map ^
  --patience=10 ^
  --eval_batch_size=1 ^
  --eval_samples=1000  ^
  --num_examples_per_epoch=2000 ^
  --num_epochs=80
</pre>

<table style="border: 1px solid #000;">
<tr>
<td>
--mode</td><td>train_and_eval</td>
</tr>
<tr>
<td>
--train_file_pattern</td><td>./train/train.tfrecord</td>
</tr>
<tr>
<td>
--val_file_pattern</td><td>./valid/valid.tfrecord</td>
</tr>
<tr>
<td>
--model_name</td><td>efficientdet-d0</td>
</tr>
<tr><td>
--hparams</td><td>"input_rand_hflip=False,image_size=512,num_classes=108,label_map=./label_map.yaml"
</td></tr>
<tr>
<td>
--model_dir</td><td>./models</td>
</tr>
<tr><td>
--label_map_pbtxt</td><td>./label_map.pbtxt
</td></tr>

<tr><td>
--eval_dir</td><td>./eval
</td></tr>

<tr>
<td>
--ckpt</td><td>../../efficientdet-d0</td>
</tr>
<tr>
<td>
--train_batch_size</td><td>4</td>
</tr>
<tr>
<td>
--early_stopping</td><td>map</td>
</tr>
<tr>
<td>
--patience</td><td>10</td>
</tr>

<tr>
<td>
--eval_batch_size</td><td>1</td>
</tr>
<tr>
<td>
--eval_samples</td><td>1000</td>
</tr>
<tr>
<td>
--num_examples_per_epoch</td><td>2000</td>
</tr>
<tr>
<td>
--num_epochs</td><td>80</td>
</tr>
</table>
<br>
<br>
<b>label_map.yaml</b>
<pre>
1: 'Advisory_speed'
2: 'Bridle_path'
3: 'Bus_lane'
4: 'Bus_stop'
5: 'Children'
6: 'Crossroads_with_a_minor_road'
7: 'Crossroads_with_priority_to_the_right'
8: 'Crosswinds'
9: 'Curve'
10: 'Cyclists'
11: 'Domestic_animals'
12: 'Emergency_lay_by'
13: 'End_of_all_previously_signed_restrictions'
14: 'End_of_minimum_speed_limit'
15: 'End_of_no_overtaking'
16: 'End_of_no_overtaking_by_heavy_goods_vehicles'
17: 'End_of_priority_road'
18: 'End_of_speed_limit'
19: 'End_of_speed_limit_zone'
20: 'Expressway'
21: 'Falling_rocks'
22: 'First_aid'
23: 'Fog'
24: 'Give_way'
25: 'Give_way_ahead'
26: 'Give_way_to_oncoming_traffic'
27: 'Go_straight'
28: 'Go_straight_or_turn_right'
29: 'Height_limit'
30: 'Ice_or_snow'
31: 'Keep_right'
32: 'Lane_configuration'
33: 'Length_limit'
34: 'Level_crossing'
35: 'Level_crossing_with_barriers_ahead'
36: 'Living_street'
37: 'Loose_surface_material'
38: 'Low_flying_aircraft'
39: 'Minimum_following_distance_between_vehicles'
40: 'Minimum_speed_limit'
41: 'Motorway'
42: 'National_border'
43: 'No_agricultural_vehicles'
44: 'No_animal_drawn_vehicles'
45: 'No_buses'
46: 'No_entry'
47: 'No_heavy_goods_vehicles'
48: 'No_mopeds'
49: 'No_motorcycles'
50: 'No_motor_vehicles'
51: 'No_motor_vehicles_except_motorcycles'
52: 'No_overtaking'
53: 'No_overtaking_by_heavy_goods_vehicles'
54: 'No_parking'
55: 'No_pedal_cycles'
56: 'No_pedestrians'
57: 'No_stopping'
58: 'No_through_road'
59: 'No_u_turn'
60: 'No_vehicles'
61: 'No_vehicles_carrying_dangerous_goods'
62: 'No_vehicles_carrying_inflammables_or_explosives'
63: 'No_vehicles_carrying_water_pollutants'
64: 'No_vehicles_pulling_a_trailer'
65: 'One_way_street'
66: 'Opening_bridge'
67: 'Other_danger'
68: 'Parking'
69: 'Parking_garage'
70: 'Pedal_cycles_only'
71: 'Pedestrians'
72: 'Pedestrians_only'
73: 'Pedestrian_crossing'
74: 'Pedestrian_crossing_ahead'
75: 'Pedestrian_zone'
76: 'Police'
77: 'Priority_over_oncoming_traffic'
78: 'Priority_road'
79: 'Restricted_parking_zone'
80: 'Roadworks'
81: 'Road_narrows'
82: 'Roundabout'
83: 'Route_for_vehicles_carrying_dangerous_goods'
84: 'Segregated_pedestrian_and_cycle_path'
85: 'Series_of_curves'
86: 'Shared_pedestrian_and_cycle_path'
87: 'Slippery_surface'
88: 'Soft_verges'
89: 'Speed_limit'
90: 'Speed_limit_zone'
91: 'Steep_ascent'
92: 'Steep_descent'
93: 'Stop'
94: 'Stop_ahead'
95: 'Taxi_stand'
96: 'Toll'
97: 'Traffic_queues'
98: 'Traffic_signals'
99: 'Trams'
100: 'Tunnel'
101: 'Turn_right'
102: 'Two_way_traffic'
103: 'Uneven_surface'
104: 'Unprotected_quayside_or_riverbank'
105: 'Weight_limit'
106: 'Weight_limit_per_axle'
107: 'Width_limit'
108: 'Wild_animals'
</pre>

<br>
<b>Training console output at epoch 80</b>
<br>
<img src="./asset/coco_metrics_console_at_epoch80_tf2.8.0_0715.png" width="1024" height="auto">
<br>
<br>
<b><a href="./projects/DE_RoadSigns/eval/coco_metrics.csv">COCO meticss</a></b><br>
<img src="./asset/coco_metrics_at_epoch80_tf2.8.0_0715.png" width="1024" height="auto">
<br>
<br>
<b><a href="./projects/DE_RoadSigns/eval/train_losses.csv">Train losses</a></b><br>
<img src="./asset/train_losses_at_epoch80_tf2.8.0_0715.png" width="1024" height="auto">
<br>
<br>

<b><a href="./projects/DE_RoadSigns/eval/coco_ap_per_class.csv">COCO ap per class</a></b><br>
<img src="./asset/coco_ap_per_class_at_epoch80_tf2.8.0_0715.png" width="1024" height="auto">
<br>
<br>
<h3>
6. Create a saved_model from the checkpoint
</h3>
 Please run the following bat file to create a saved model from a chekcpoint in models folder.
<pre>
2_create_saved_model.bat
</pre>
, which is the following:
<pre>
python ../../SavedModelCreator.py ^
  --runmode=saved_model ^
  --model_name=efficientdet-d0 ^
  --ckpt_path=./models  ^
  --hparams="image_size=512x512,num_classes=108" ^
  --saved_model_dir=./saved_model
</pre>

<table style="border: 1px solid #000;">
<tr>
<td>--runmode</td><td>saved_model</td>
</tr>

<tr>
<td>--model_name </td><td>efficientdet-d0 </td>
</tr>

<tr>
<td>--ckpt_path</td><td>./models</td>
</tr>

<tr>
<td>--hparams</td><td>"image_size=512x512,num_classes=108"</td>
</tr>

<tr>
<td>--saved_model_dir</td><td>./saved_model</td>
</tr>
</table>

<br>
<br>
<h3>
7. Inference FR_RoadSigns by using the saved_model
</h3>
 Please run the following bat file to infer the roadsigns by using the saved_model:
<pre>
</pre>
, which is the following:
<pre>
python ../../SavedModelInferencer.py ^
  --runmode=saved_model_infer ^
  --model_name=efficientdet-d0 ^
  --saved_model_dir=./saved_model ^
  --min_score_thresh=0.4 ^
  --hparams="label_map=./label_map.yaml" ^
  --input_image=./realistic_test_dataset/*.jpg ^
  --classes_file=./classes.txt ^
  --ground_truth_json=./realistic_test_dataset/annotation.json ^
  --output_image_dir=./realistic_test_dataset_outputs
</pre>

<table style="border: 1px solid #000;">
<tr>
<td>--runmode</td><td>saved_model_infer </td>
</tr>

<tr>
<td>--model_name</td><td>efficientdet-d0 </td>
</tr>

<tr>
<td>--saved_model_dir</td><td>./saved_model </td>
</tr>

<tr>
<td>--min_score_thresh</td><td>0.4 </td>
</tr>

<tr>
<td>--hparams</td><td>"label_map=./label_map.yaml"</td>
</tr>

<tr>
<td>--input_image</td><td>./realistic_test_dataset/*.jpg</td>
</tr>

<tr>
<td>--classes_file</td><td>./classes.txt</td>
</tr>
<tr>
<td>--ground_truth_json</td><td>./realistic_test_dataset/annotation.json</td>
</tr>

<tr>
<td>--output_image_dir</td><td>./realistic_test_dataset_outputs</td>
</tr>
</table>
<br>
<h3>
8. Some inference results of FR RoadSigns
</h3>

<img src="./projects/DE_RoadSigns/realistic_test_dataset_outputs/de_roadsigns_1000.jpg" width="1280" height="auto"><br>
<a href="./projects/DE_RoadSigns/realistic_test_dataset_outputs/de_roadsigns_1000.jpg_objects.csv">roadsigns_1.jpg_objects.csv</a><br>

<img src="./projects/DE_RoadSigns/realistic_test_dataset_outputs/de_roadsigns_1020.jpg" width="1280" height="auto"><br>
<a  href="./projects/DE_RoadSigns/realistic_test_dataset_outputs/de_roadsigns_1020.jpg_objects.csv">roadsigns_2.jpg_objects.csv</a><br>

<img src="./projects/DE_RoadSigns/realistic_test_dataset_outputs/de_roadsigns_1030.jpg" width="1280" height="auto"><br>
<a  href="./projects/DE_RoadSigns/realistic_test_dataset_outputs/de_roadsigns_1030.jpg_objects.csv">roadsigns_3.jpg_objects.csv</a><br>

<img src="./projects/DE_RoadSigns/realistic_test_dataset_outputs/de_roadsigns_1040.jpg" width="1280" height="auto"><br>
<a  href="./projects/DE_RoadSigns/realistic_test_dataset_outputs/de_roadsigns_1040.jpg_objects.csv">roadsigns_4.jpg_objects.csv</a><br>

<img src="./projects/DE_RoadSigns/realistic_test_dataset_outputs/de_roadsigns_1050.jpg" width="1280" height="auto"><br>
<a  href="./projects/DE_RoadSigns/realistic_test_dataset_outputs/de_roadsigns_1050.jpg_objects.csv">roadsigns_5.jpg_objects.csv</a><br>

<img src="./projects/DE_RoadSigns/realistic_test_dataset_outputs/de_roadsigns_1060.jpg" width="1280" height="auto"><br>
<a  href="./projects/DE_RoadSigns/realistic_test_dataset_outputs/de_roadsigns_1060.jpg_objects.csv">roadsigns_6.jpg_objects.csv</a><br>

<img src="./projects/DE_RoadSigns/realistic_test_dataset_outputs/de_roadsigns_1070.jpg" width="1280" height="auto"><br>
<a  href="./projects/DE_RoadSigns/realistic_test_dataset_outputs/de_roadsigns_1070.jpg_objects.csv">roadsigns_7.jpg_objects.csv</a><br>

<img src="./projects/DE_RoadSigns/realistic_test_dataset_outputs/de_roadsigns_1080.jpg" width="1280" height="auto"><br>
<a  href="./projects/DE_RoadSigns/realistic_test_dataset_outputs/de_roadsigns_1080.jpg_objects.csv">roadsigns_8.jpg_objects.csv</a><br>

<img src="./projects/DE_RoadSigns/realistic_test_dataset_outputs/de_roadsigns_1090.jpg" width="1280" height="auto"><br>
<a  href="./projects/DE_RoadSigns/realistic_test_dataset_outputs/de_roadsigns_1090.jpg_objects.csv">roadsigns_9.jpg_objects.csv</a><br>

<img src="./projects/DE_RoadSigns/realistic_test_dataset_outputs/de_roadsigns_1099.jpg" width="1280" height="auto"><br>
<a  href="./projects/DE_RoadSigns/realistic_test_dataset_outputs/de_roadsigns_1099.jpg_objects.csv">roadsigns_10.jpg_objects.csv</a><br>

<h3>9. COCO metrics of inference result</h3>
The 3_inference.bat computes also the COCO metrics(f, map, mar) to the <b>realistic_test_dataset</b> as shown below:<br>

<a href="./projects/FR_RoadSigns/realistic_test_dataset_outputs/prediction_f_map_mar.csv">prediction_f_map_mar.csv</a>

<br>
<img src="./asset/coco_metrics_console_test_dataset_at_epoch80_tf2.8.0_0715.png" width="740" height="auto"><br>

