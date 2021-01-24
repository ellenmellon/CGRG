# A-Controllable-Model-of-Grounded-Response-Generation



## folders:
./data folder contains sample raw_data.txt (processed after the raw reddit data).


## env setup:
requires python3.6
requirements.txt was created by pip freeze, you probably don't need that many packages.
download and unzip the [folder](https://drive.google.com/file/d/1IjpVacKkafuALM9dlOI5chUaQdEa9jOZ/view?usp=sharing) containing the pretrained model under ./src folder. 

commands:
> cd src
> bash run.sh


## format of data/raw_data.txt (so that you can create your own):

data examples are seperated by an empty line
each data example has the following format:

instance index <br>
previous utterances separated by <t1>, <t2>, ...  <br>
target response  <br>
grounding sentence s1  <br>
control phrase in s1  <br>
grounding sentence s2  <br>
control phrase in s2   <br>
...   <br>
...  <br>
