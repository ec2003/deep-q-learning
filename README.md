# Deep Q-Learning

## How to use
Python Version 3.11.5
 ### 1. Install required libraries
Windows:
```bat
pip install -r requirements.txt
```
Mac:
```bat
pip3 install -r requirements.txt
```
### 2. Run the program
#### Default options
For both training and testing:
```bat
start run_test_and_train.bat
```
For training only:
```bat
start run_train.bat
```
For testing only:
```bat
start run_test.bat
```
#### Run with manual configurations
Default:
```bat
python main.py --train --test --train_episodes=10000 --test_episodes=10 
```
We supports these arguments for running with configurations:

 - `--train` to enable training, `--no-train` to disable training.
 - `--test` to enable testing, `--no-test` to disable testing.
 - `--train_episodes=<int>`: choose the number of episodes for training.
 - `--test_episodes=<int>`: choose the number of episodes for testing.
