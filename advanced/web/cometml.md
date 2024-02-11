## Comet.ml

### Log a distributed training run with multiple Experiments

To capture model metrics and system metrics (GPU/CPU usage, RAM, and so on) from each machine while running distributed training, we recommend creating **an Experiment object per GPU process** and grouping these experiments under a user provided run ID.
