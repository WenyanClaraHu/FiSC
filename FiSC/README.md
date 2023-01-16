# FiSC model: Forecasting fine-grained sensing coverage model
Two parts are introduced, consisting of the training stage and the implementing stage.
## Training
### Dataset (data_80_80_2000_training)
In this demo, the study area is Shanghai, China. The spatial granularity is 80*80. The training dataset is extracted based on trajectories of 2000 vehicles (several examples are provided in the folder **data_80_80_2000_training**). The input dataset is a 12 by 6400 matrix based on 12 images (representing the number of vehicles, temporal granularity, proportion of selected vehicles in different districts, proportion of selected vehicles in different districts during night hours, proportion of selected vehicles in different districts during peak hours, proportion of selected vehicles in different districts during day hours, driving time per day, driving time during night hours, driving time during peak hours, driving time during day hours, traffic flow and the number of bus stations, respectively), while the output is a 1 by 6400 matrix (representing the sensing coverage ratio in each cell).
### Requirements
* **tensorflow**
* **numpy**
* **argparse**
* **tqdm**
### Run the demo
train_addingMask_validation.py
## implementing
### Dataset (data_80_80_2000_implementing)
In this demo, the study area is Shanghai, China. The spatial granularity is 80*80. The input dataset (several examples are provided in the folder **data_80_80_2000_implementing**) is calculated based on the task requirement and coarse-grained information of candidate vehicles. Meanwhile, the trained FiSC is used to forecast fine-grained sensing coverage.
### Requirements
* **tensorflow**
* **numpy**
* **argparse**
* **tqdm**
### Material
* [The trained model](https://unimelbcloud-my.sharepoint.com/:f:/r/personal/wenyan1_student_unimelb_edu_au/Documents/train_log_80_2000veh?csf=1&web=1&e=zmSZQY)  
### Run the demo
prediction.py
