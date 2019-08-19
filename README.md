# openpilot-lane-detection
an end to end lane detection based on openpilot model

## output
![test.gif](https://github.com/CodingSheep1229/openpilot-lane-detection/blob/master/src/demo/test.gif?raw=true)

## Dependency
1. SNPE
2. opencv
3. Eigen (c++)
4. numpy


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

## Reference
[Commaai](https://github.com/commaai/openpilot)
