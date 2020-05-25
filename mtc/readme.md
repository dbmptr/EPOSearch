## Experiment on Multi Task Classification

First run the training scripts:
```
python individual_train.py
python linscalar_train.py
python epo_train.py
python pmtl_train.py
```

This will create `.pkl` files in the `results` folder. Then use `display_result.py` to obtain the figures.