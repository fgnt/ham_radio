# Segmented RNN

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/fgnt/lazy_dataset/blob/master/LICENSE)

If you want to train a neural network for speech activity detection 
on the ham_radio database follow these steps:
1. Clone this repository and install it with pip
1. Download the database with 
```bash
    wget -qO- https://zenodo.org/record/4247491/files/ham_rdaio.tar.gz.parta{a,b,c,d,e} \
	| tar -C /PATH/TO/HAM_RADIO_DB/ -zx --checkpoint=10000 --checkpoint-action=echo="%u/5530000 %c"
``` 
where `/PATH/TO/HAM_RADIO_DB` has to be replaced with the chosen 
database directory
1. Set the variable ```HAM_RADIO_JSON_PATH``` to the file name the database json
should be written to
    ```export HAM_RADIO_JSON_PATH=/PATH/TO/JSON```
1. Create a database json with
```bash
python -m ham_radio.database.ham_radio.create_json \
    with database_path=/PATH/TO/HAM_RADIO_DB
```
1. Set a directory to which to write all models with
    ```export STORAGE_ROOT=/PATH/TO/MODEL_DIR```
1. Start a training with:
    ```bash
    python -m ham_radio.train with cnn
    ``` 
 The trained model and the event files are written to 
 `/PATH/TO/MODEL_DIR/ham_radio/SADModel_{number_of_train_runs}`
 For more information about the training script and the event files visit 
 our [padertorch repository](https://github.com/fgnt/padertorch)
 
 If you want to reduce the required space for the gpu you can reduce 
 the batch size by adding ```provider_opts.batch_size=4``` or any other 
 value for the batch size.
   
 If you want to use a simple RNN structure instead of the  
 RNN you can replace ```cnn```  with ```rnn```
 Most paramters are adjustable in a similar fashion.