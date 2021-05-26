# A-Controllable-Model-of-Grounded-Response-Generation


## env setup:
Run:
1. `conda env create -f cgrg.yml`
2. `conda activate cgrg`
3. `bash setup.sh`

## Data:
1. Download Reddit data from the [original git repo](https://github.com/qkaren/converse_reading_cmr).
2. Put the unzipped folder under ./data/dstc and name as ./data/dstc/raw


## Models:
1. Download and unzip the [folder](https://drive.google.com/file/d/1IjpVacKkafuALM9dlOI5chUaQdEa9jOZ/view?usp=sharing) containing the pretrained GPT2 model under ./src folder. 
2. Trained CGRG model on Reddit can be found [here](https://drive.google.com/file/d/16dsafcAuGSU_mG9lk_pH87sreSkAYC_Q/view?usp=sharing)


## Commands:
1. `cd prepare_data`
2. `bash prepare_data.sh`
3. `cd src`
4. `bash run.sh`
