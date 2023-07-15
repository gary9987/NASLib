# NASLIB
## For the original README please refer to [NASLIB](https://github.com/automl/NASLib)
## This compare our work with LS, RS and RE
### Pre-request
#### NASBench101
- Please follow the instructions of the original NASLIB repo [NASLIB](https://github.com/automl/NASLib#queryable-benchmarks)
#### NASBench201
- Download NASBench201 dataset cache [Download link](https://www.dropbox.com/scl/fi/pq96opeyd4dnufdifznyb/nb201_datasets_cache.zip?rlkey=r2z0etza5aujuucxy2waep14v&dl=1), and put it to the path `NASLib/examples/datasets`
### Step
- modify the config files in `naslib\defaults`, the main part we need to modify is `dataset`
  - nb101_ls.yaml
  - nb101_re.yaml
  - nb201_ls.yaml
  - ...
- run the experiment script `examples/demo.py`, for example:
  ```
  python demo.py --gpu 0 --config-file ../naslib/defaults/nb201_rs.yaml
  ```