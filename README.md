# Distributed Training for Dummies

This repository accompanies the blog post: [Distributed Training for Dummies](https://www.pgupta.info/blog/2025/07/distributed-training-0/).

It contains code and hands-on examples to help you understand and experiment with the core concepts of distributed training in deep learning, including:
- **Data Parallelism**
- **Tensor Parallelism**
- **Pipeline Parallelism**

Follow along with the blog to explore and play with these distributed training techniques. Each directory in this repo corresponds to a step or concept discussed in the blog, making it easy to map code to theory.

## Setup

You can run the code either locally (with Python) or using Docker.

### 1. Clone the repository

```bash
git clone https://github.com/pg2455/distributed_training_for_dummies.git
cd distributed_training_for_dummies
```

### 2. Install dependencies (for local/torchrun mode)

```bash
pip install -r requirements.txt
```

## Running Distributed Training

You can run the scripts in two ways:

### **A. Docker Compose (Simulate Multiple Machines)**
- Simulates multiple nodes/GPUs using containers.
- Communication is managed by us in the code.
- You can set the `RUN_MODE` environment variable to select the example to run:
  - `comm` — Communication primitives demo
  - `train` — Simple training demo
  - `data_parallel_naive`
  - `data_parallel_naive_grad_accumulation`
  - `data_parallel_overlap`
  - `data_parallel_bucket_overlap`
  - `tensor_parallel`

**Example:**
```bash
RUN_MODE=data_parallel_naive docker-compose up --build
```
Check `entrypoint.sh` to see what commands are run in each of these `RUN_MODE`s. 

### **B. torchrun (PyTorch Native Multi-Process Launch)**
- PyTorch manages process launch and communication.
- Useful for running on a single machine with multiple processes, or across nodes.

**Example:**
```bash
torchrun --nproc_per_node=2 tensor_parallel train.py
```

For more advanced distributed setups:
```bash
torchrun --nnodes=2 --node_rank=0 --master_addr="<master_ip>" --master_port=12355 data_parallel/train_naive.py
```




## Customization
- Adjust `WORLD_SIZE`, `RANK`, and other distributed settings in `docker-compose.yaml` as needed.
- For more details on each script, see comments in the code and follow along with the blog.
