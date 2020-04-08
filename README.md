# Segmented RNN

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/fgnt/lazy_dataset/blob/master/LICENSE)

To start a training on one of the example databases
 follow these steps:
1. Download the Database you want to train on
1. Clone this repository and install it with pip as editable
1. Create a database json
    * For the Ham-Radio-Database use the following code which will by default 
    will write a json file to ./ham_radio.json:
   ```bash
    python -m segmented_rnn.database.ham_radio.create_json \
        -j /PATH/TO/JSON -db /PATH/TO/HAM_RADIO_DB
    ```           
    *
1. Set the variable ```HAM_RADIO_JSON``` to your written json file:\
    ```export HAM_RADIO_JSON=/PATH/TO/JSON```
2. Set a directory to which to write all models with
    ```export MODEL_DIR=/PATH/TO/MODEL_DIR```
1. Start a training with:
    ```bash
    python -m segmented_rnn.train with ham_radio segmented_rnn
    ``` 
 
 If you want to reduce the required space for the gpu you can reduce
  the batch size by adding ```provider_opts.batch_size=4``` or any other
   value for the batch size.
   
 If you want to use a simple RNN or CNN structure instead of the
 segmented RNN you can replace ```segmented_rnn```  with ```rnn``` or ```cnn```
 
 Most paramters are adjustable in a similar fashion.