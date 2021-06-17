# A-Controllable-Model-of-Grounded-Response-Generation


## Environment Setup:
Run:
1. `conda env create -f cgrg.yml`
2. `conda activate cgrg`
3. `bash setup.sh`

## Data and Model Preparation:
1. Download Reddit data from the [original git repo](https://github.com/qkaren/converse_reading_cmr).
2. Put the unzipped folder under `./data/dstc` and name as `./data/dstc/raw`
3. You can skip the above two steps if using the [preprocessed files](https://drive.google.com/file/d/1Nj9dveY6s666KRB0yhBtGQ3M7eWfNOJ1/view?usp=sharing). Unzip it and put under `./data`. It contains a toy test file. Note that the preprocessed files we provide are based on an earlier version of the Reddit dataset, which is slightly differently from the version provided in the above github repo.
4. Download and unzip the [folder](https://drive.google.com/file/d/1IjpVacKkafuALM9dlOI5chUaQdEa9jOZ/view?usp=sharing) containing the pretrained GPT2 model under `./src` folder. 

You can create your own processed data in the same format as files in the link of step 3. Here is the format: <br>
instance index (order not required) <br>
previous utterances <br>
target response <br>
grounding sentence s1 <br>
control phrase in s1 <br>
grounding sentence s2 <br>
control phrase in s2 <br>
... <br>
... <br>


## Training and Inference:
If you chose to use the preprocessed data above in step 3 above, you can skip step 2 below. Step 3 would take some time.
1. `cd prepare_data`
2. `bash preprocess.sh`
3. `bash prepare_model_inputs.sh`
4. `cd src`
5. `bash run.sh`


## Evaluation
See requirements in the README file under `./eval`. Run:
1. `cd eval`
2. `python create_eval_files.py YOUR_OUTPUT_FILE_FROM_STEP_5_ABOVE`
3. `python dstc.py pred.txt -rf ref.txt`


## Cite
```
@inproceedings{wu-etal-2021-cgrg,
    author = "Wu, Zeqiu and Galley, Michel and Brockett, Chris and Zhang, Yizhe and Gao, Xiang and Quirk, Chris and Koncel-Kedziorski, Rik and Gao, Jianfeng and Hajishirzi, Hannaneh and Ostendorf, Mari and Dolan, Bill",
    title = "A Controllable Model of Grounded Response Generation",
    booktitle = "AAAI",
    year = "2021",
    month = "January",
}
```
