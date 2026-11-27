# Model card: SynapseMNIST3D, Kinetics-pretrained R3D-18

## Summary
A baseline for the SynapseMNIST3D paper: torchvision's R3D-18, pretrained on Kinetics-400 video,
fine-tuned with the same recipe as `synapsemnist3d-resnet18` to classify 64-voxel electron
microscopy volumes as excitatory or inhibitory synapses. Results come from the published runs'
`metrics.json`.

## Intended use
Research only. Not a medical device. It exists to compare against the from-scratch 3D ResNet-18.

## Data
SynapseMNIST3D (MedMNIST v2), CC BY 4.0, official split at 64 voxels. MedMNIST has no patient
IDs, so the bootstrap resamples volumes.

## Training
Config: `config.yaml`. The one-channel volume is repeated to three channels and normalized with
Kinetics-400 statistics; depth plays the role of time. Commit, hardware and time: each run's
`run.json`.

## Limitations and biases
The recipe was tuned for training from scratch, not for fine-tuning a pretrained network.

## Reproduce
```bash
uv run opendiagnostics fit -c models/synapsemnist3d-r3d18/config.yaml --seed_everything=<seed>
```
