import os
import argparse
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


# data parallel (distribute model across two processes)
# 1. construct micrbatches, accumulate gradients, & process them in parallel
# 2. ZeRO - shard optimizer states, gradients, model parameters


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, size=1000):
        self.data = torch.randn(size, 10)
        self.target = F.sigmoid(self.data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]
    
def print_if_rank_0(rank, message):
    if rank == 0:
        print(message)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for toy model")
    parser.add_argument("--grad_acc_steps", type=int, default=1)
    args = parser.parse_args()

    grad_acc_steps = args.grad_acc_steps
    NUM_STEPS_PER_EPOCH = 2 # this is the number of micro-batches per epoch
    
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    master_addr = os.environ['MASTER_ADDR']
    master_port = os.environ['MASTER_PORT']

    print_if_rank_0(rank, f"grad_acc_steps: {grad_acc_steps}")
    print_if_rank_0(rank, f"world_size: {world_size}")

    dist.init_process_group(
        backend='gloo',
        init_method=f'tcp://{master_addr}:{master_port}',
        rank=rank,
        world_size=world_size
    )

    dataset = SimpleDataset()
    
    # we need a distributed dataloader so that each process gets a unique, non-overlapping subset of data. 
    # Each process will work on a different part of the dataset. 
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=10, 
        sampler = DistributedSampler(
            dataset, 
            num_replicas=world_size, 
            rank=rank,
            shuffle=True, # in non-distributed setting, this is specified in the dataloader.
            # in distributed setting, at the start of each epoch, shuffle is done via `sampler.set_epoch(epoch)`
        )
    )
    
    # 1. Share the same model across ranks.
    # each rank has its own copy of the same model weights.
    model = Model() 
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    # 1.b Another way to have the same model across ranks is to set the same seed for each rank.

    # each rank holds its own optimizer state. 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    dp_process_group = dist.new_group([0, 1])
    dp_world_size = dist.get_world_size(group=dp_process_group)
    print_if_rank_0(rank, f"dp_world_size: {dp_world_size}")

    # This is naive data parallel. 
    # here, after gradients of the entire model are computed, we sum and sync them across ranks.
    # Computation is done and then communication is done after computation is complete. 
    for epoch in range(2):
        dataloader.sampler.set_epoch(epoch) # shuffle the data at the start of each epoch
        dataloader_iter = iter(dataloader)
        
        for batch_idx in range(NUM_STEPS_PER_EPOCH):
            acc_loss = 0.0
            optimizer.zero_grad()
            for i in range(grad_acc_steps):
                batch = next(dataloader_iter)
                data, target = batch # each rank gets a unique, non-overlapping subset of data. 

                result = model(data)
                loss = F.mse_loss(result, target)
                loss.backward() # it accumulates gradients across micro-batches 
                acc_loss += loss.item()
                print(f"Rank {rank}, Epoch {epoch}, Batch {batch_idx}, Microstep: {i}, Local loss: {loss}")
            
            print(f"Rank {rank}, Epoch {epoch}, Batch {batch_idx}, Accumulated loss: {acc_loss}")
            dist.barrier() # wait for all ranks to finish the current batch
            # Important to sync before gradients are accumulated 

            # 2. This constitutes Data Parallelism. Accumulate and sync gradients across ranks.
            # sum gradients across ranks and sync gradients across ranks
            for param in model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                    param.grad /= dp_world_size

            print(f"Rank {rank} grad: {model.linear.weight.grad[-1]}") # should be the same across ranks
            optimizer.step()
            print(f"Rank {rank} param: {model.linear.weight[-1]}") # should be the same across ranks

    dist.destroy_process_group()


