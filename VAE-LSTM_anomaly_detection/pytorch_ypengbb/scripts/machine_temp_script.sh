# need to add JSON file path in the argparser (specified in utils.py -> get_args())

python preprocess_NAB_data.py -c pytorch_NAB_config.json \
&& \
python pytorch_anomaly_detection_main.py -c pytorch_NAB_config.json