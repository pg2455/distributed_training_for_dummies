import os
import argparse
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


# data parallel (distribute model across two processes)
# 1. construct micrbatches, accumulate gradients, & process them in parallel
# 2. ZeRO - shard optimizer states, gradients, model parameters
def print_if_rank_0(rank, message):
    if rank == 0:
        print(message)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(10, 1)

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, size=1000):
        self.data = torch.randn(size, 10)
        self.target = torch.sum(F.sigmoid(self.data), dim=1, keepdim=True)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]
    

class DataParallel(torch.nn.Module):
    """
    We use buckets to track gradients and optimizer states.
    """
    def __init__(self, module, process_group):
        super().__init__()
        self.module = module
        self.dp_process_group = process_group
        self.dp_world_size = dist.get_world_size(group=self.dp_process_group)
        self.require_backward_sync = False # Set to False while accumulating gradients. Set to True after gradients are accumulated.
        self.register_backward_hook(self._allreduce_grads)
    
    def forward(self, x):
        return self.module(x)
    
    def backward(self, x):
        return self.module.backward(x)
    
    def register_backward_hook(self, hook):
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_hook(hook)

    def _allreduce_grads(self, grad):
        """
        This is the hook that is called when the gradients are available after the backward pass.
        It is called for each parameter in the model. 
        This way we can overlap computation and communication.
        """
        if self.require_backward_sync:
            dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=self.dp_process_group)
            grad /= self.dp_world_size


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

    dp_process_group = dist.new_group([0, 1])
    dp_world_size = dist.get_world_size(group=dp_process_group)
    print_if_rank_0(rank, f"dp_world_size: {dp_world_size}")

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

    model = DataParallel(model, dp_process_group)

    # 1.b Another way to have the same model across ranks is to set the same seed for each rank.

    # each rank holds its own optimizer state. 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # This is data parallel where we overlap computation and communication.
    # Computation requires forwards and backwards passes.
    # Communication requires summing gradients and syncing them across ranks.
    # We do so as soon as the gradients of parameters are computed. Therefore, we need to add a hook to the backward pass of the param. 
    # This hook initiates communication as soon as the gradients are available.
    for epoch in range(2):
        dataloader.sampler.set_epoch(epoch) # shuffle the data at the start of each epoch
        dataloader_iter = iter(dataloader)
        
        for batch_idx in range(NUM_STEPS_PER_EPOCH):
            acc_loss = 0.0
            optimizer.zero_grad()
            for i in range(grad_acc_steps):
                model.require_backward_sync = i == grad_acc_steps - 1 # set to True at the last step.
                batch = next(dataloader_iter)
                data, target = batch # each rank gets a unique, non-overlapping subset of data. 

                print(f"Rank {rank} data: {data.shape} target: {target.shape}")
                result = model(data)
                loss = F.mse_loss(result, target)
                loss.backward() 
                acc_loss += loss.item()
                print(f"Rank {rank}, Epoch {epoch}, Batch {batch_idx}, Microstep: {i}, Local loss: {loss}")

            print(f"Rank {rank}, Epoch {epoch}, Batch {batch_idx}, Accumulated loss: {acc_loss}")
            dist.barrier() # wait for all ranks to finish the current batch

            print(f"Rank {rank} grad: {model.module.linear.weight.grad[-1]}") # should be the same across ranks
            optimizer.step()
            print(f"Rank {rank} param: {model.module.linear.weight[-1]}") # should be the same across ranks

    dist.destroy_process_group()


