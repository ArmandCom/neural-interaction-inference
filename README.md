# Inferring Relational Potentials in Interacting Systems

This is the pytorch code for the paper [Inferring Relational Potentials in Interacting Systems](https://openreview.net/forum?id=Iwt7oI9cNb).


https://github.com/ArmandCom/neural-interaction-inference/assets/33665810/4aa43228-1837-4230-a55d-bf9b888764ee

## Datasets

For Springs and/or Charge datasets execute the simulation code in folder "./data".
```
python generate_dataset.py --simulation <charged, springs-strong> --n-balls 5 --datadir ./datasets/
```

## Training NIIP
```
python train.py --train --num_steps=5 --num_steps_test 5 --num_steps_end 5 --ns_iteration_end 200000 --cd_and_ae --mse_all_steps --step_lr=0.2 --dataset={dataset_name} --batch_size=20 --latent_dim=64 --latent_hidden_dim 256 --filter_dim 256 --num_fixed_timesteps 1 --num_timesteps 70 --forecast 21 --n_objects 5 --pred_only --normalize_data_latent --ensembles 2 --factor --masking_type random --logname filters_256 --cuda --gpu_rank <gpu_id>
```

## Forecasting

To download the pretrained models, access this [link](https://www.dropbox.com/scl/fo/3t8arx3kzgt57h58g495t/h?rlkey=2xlozc6sc78bqr3f8n164raxm&dl=0) and save it in your machine. [Charged](https://www.dropbox.com/scl/fi/inkm3m7etdu46hpbq69qk/charged_weights.zip?rlkey=9vzlsoxjq0qh22ofuc8t4pwlp&dl=0), [Springs](https://www.dropbox.com/scl/fi/buwarglwev61etwbubxrp/springs_weights.zip?rlkey=hw7crji0hyzame8c0k0diq394&dl=0).

For each dataset, the main test command is:
```
python train.py TEST_FLAGS
```
```
TEST_FLAGS =
--num_steps_test 5 
--step_lr=0.2 
--dataset=<dataset_name>
--batch_size=20 
--latent_dim=64 
--num_fixed_timesteps 1 
--factor 
--num_timesteps 70 
--forecast 21 
--n_objects 5 
--pred_only 
--ensembles 2 
--masking_type random 
--logname filters_256 
--latent_hidden_dim 256 
--filter_dim 256 
--resume_name <path/to/downloaded/weights> 
--normalize_data_latent (for Charged)
--resume_iter -1
--cuda 
--gpu_rank <gpu_id>
```

# Manipulate Trajectories

With the following commands we can add new potentials at test time.
As an example, for the Charged dataset, we can use "avoid_area" with a magnitude of 1e-1.

```
python train.py TEST_FLAGS
--test_manipulate
--new_energy <avoid_area, velocity, attraction, attract_to_center>
--new_energy_magnitude <1e-1 (e.g.)>
```

# Citing our Paper

If you find our work useful for your research, please consider citing 

``` 
@inproceedings{Comas2023InferringRP,
  title={Inferring Relational Potentials in Interacting Systems},
  author={Armand Comas Massague and Yilun Du and Christian Fernandez and Sandesh Ghimire and Mario Sznaier and Joshua B. Tenenbaum and Octavia I. Camps},
  booktitle={International Conference on Machine Learning},
  year={2023},
 }
```
