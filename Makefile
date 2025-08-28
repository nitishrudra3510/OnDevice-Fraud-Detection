VENV=/Users/nitishkumar/Documents/GitHub/Multi-Agent/.venv
PY=$(VENV)/bin/python3
PIP=$(VENV)/bin/pip

.PHONY: venv install data features preprocess train optimize tune run ui all clean

venv:
	python3 -m venv $(VENV) || true
	. $(VENV)/bin/activate; $(PIP) install --upgrade pip wheel
	. $(VENV)/bin/activate; $(PIP) install -r requirements.txt || true

install:
	. $(VENV)/bin/activate; $(PIP) install numpy pandas scikit-learn tensorflow==2.20.0 kivy || true

data:
	$(PY) 2_Datasets/Synthetic_Data_Generator.py --num_users 100 --num_records 2500 --seed 123

features:
	$(PY) 3_Feature_Engineering/feature_extraction.py

preprocess:
	$(PY) 3_Feature_Engineering/preprocessing.py

train:
	$(PY) 4_Models/one_class_svm.py
	$(PY) 4_Models/isolation_forest_model.py
	$(PY) 4_Models/autoencoder_model.py
	$(PY) 4_Models/movement_cnn.py

optimize:
	$(PY) 5_OnDevice_Optimization/convert_to_tflite.py

tune:
	$(PY) 4_Models/enroll_and_tune.py --weight_candidates 0.1,0.2,0.3,0.4 --threshold_candidates 0.4,0.5,0.6,0.7

run:
	$(PY) 4_Models/multi_agent_system.py

ui:
	$(PY) 7_Demo_App/mobile_integration/kivy_demo.py

all: venv install data features preprocess train optimize tune run

clean:
	rm -rf 3_Feature_Engineering/.artifacts 4_Models/.models 4_Models/.config 5_OnDevice_Optimization/.tflite

