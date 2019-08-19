# openpilot-lane-detection
an end to end lane detection based on openpilot model

## Dependency
1. SNPE
2. opencv
3. 

## Run
convert png to yuv raw files
```
python convert.py
```

run openpilot model
```
snpe-net-run --container driving_model.dlc --input_list data/raw_list.txt
```

post-processing is in src/demo.ipynb