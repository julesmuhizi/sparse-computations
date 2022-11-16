# Anomaly detection with HLS4ML


## Training
Instructions on how to train and export the AD08 model can be found in [training/README.md](training/README.md).
Here also additional instructions for the bin file generation can be found.

### Model Architecture
![Alt text](training/ad08_model.png?raw=true "Title")

## Convert and Synthesize
```bash
cd inference
python convert.py -c <model_config>.yml
```
- To build for the Pynq-Z2, use the `ad08_pynq.yml` config file.
- To build for the Arty A7-100T, use the `ad08_arty_<accuracy|power>.yml` config file, choosing `accuracy` if you want to run accuracy/latency benchmarks, or `power` if you want to run energy benchmarks
