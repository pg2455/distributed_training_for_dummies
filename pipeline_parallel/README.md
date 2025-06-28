

How to run the code

```bash
torchrun --nproc_per_node=3 train_afab.py  --grad_acc_steps 2 --pseudo_computation_time 0.5
```