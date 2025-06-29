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
    

class Bucket:
    """
    A bucket is a collection of parameters that are all reduced together.
    Once the synchronization of gradients is required and the params in the bucket have their gradients ready, 
    all the parameters in the bucket synchronize their gradients using allreduce. 
    """
    def __init__(self, params, grads, process_group):
        self.params = set(params)
        self.grads = grads
        self.process_group = process_group
        self.process_group_size = dist.get_world_size(group=self.process_group)

        # Set of parameters that have their gradients ready for synchronization.
        self.params_with_grad_ready = set()
        self.handle = None
        print(f"******* Bucket initialized with {len(self.params)}: {[param.shape for param in self.params]}")

    def mark_param_as_ready(self, param):
        self.params_with_grad_ready.add(param)
        if len(self.params_with_grad_ready) == len(self.params):
            self.sync_gradients()

    def sync_gradients(self):
        print(f"****** Syncing gradients. {[param.shape for param in self.params]} params with grad ready")
        assert self.handle is None, "You should not call sync_gradients twice"
        # Note: we do this asynchronously, so we capture the coroutine handle. 
        self.grads.div_(self.process_group_size)
        self.handle = dist.all_reduce(self.grads, op=dist.ReduceOp.SUM, group=self.process_group, async_op=True)

    def wait(self):
        assert self.handle is not None, "You should launch an allreduce operation before waiting for it to finish"
        self.handle.wait()

    def reset(self):
        self.params_with_grad_ready.clear()
        self.handle = None
        self.grads.zero_()


class DataParallelBucket(torch.nn.Module):
    """
    We use buckets to track gradients and optimizer states.
    """
    def __init__(self, module, process_group):
        super().__init__()
        self.module = module
        self.dp_process_group = process_group
        self.dp_world_size = dist.get_world_size(group=self.dp_process_group)
        self.require_backward_sync = False # Set to False while accumulating gradients. Set to True after gradients are accumulated.

        self.bucket_size = 50 # number of parameters in a bucket.
        self.buckets = []
        self._param_to_bucket_location = {}
        self._initialize_buckets()
        self.register_backward_hook()

        # Once the accumulation is complete, the autograd engine needs to call self._post_backward after the backward pass through the entire model. 
        self._post_backward_callback_set = False

    def _initialize_buckets(self):
        # We divide the parameters into buckets 
        # Each param exist in a bucket. 
        cur_bucket_size = 0
        cur_bucket_idx = 0
        for param in self.module.parameters():
            if not param.requires_grad:
                continue 
            
            # Note: this is not strict bucketing, but it is ok for the toy example. 
            self._param_to_bucket_location[param] = (cur_bucket_size, cur_bucket_size + param.numel(), cur_bucket_idx)

            cur_bucket_size += param.numel()
            if cur_bucket_size > self.bucket_size:
                cur_bucket_idx += 1 # start a new bucket
                cur_bucket_size = 0

        # Gather information about the bucket sizes and the parameters in each bucket
        buckets_to_params = [[] for _ in range(cur_bucket_idx + 1)]
        for param, (_, end, idx) in self._param_to_bucket_location.items():
            buckets_to_params[idx].append(param)

        # Create buckets. We need to create a unified tensor that can hold gradients for all the parameters. 
        # This is needed because during all reduce, we sum and sync this unified tensor.
        # We maintain a reference (param.main_grad) to the portion of the tensor that will be used to represent the gradients of those specific parameters.
        self.grad_list = []
        for params in buckets_to_params:
            n_params = sum(param.numel() for param in params)
            grads = torch.zeros(n_params, dtype=torch.float32, device='cpu')
            self.grad_list.append(grads)
            self.buckets.append(Bucket(params, grads, self.dp_process_group))

        # Create gradient views into param.main_grad. This is what we will use to accumulate gradients.
        for param in self.module.parameters():
            if param.requires_grad:
                data_start_idx, data_end_idx, bucket_idx = self._param_to_bucket_location[param]
                param.main_grad = self.grad_list[bucket_idx][data_start_idx:data_end_idx].view(param.shape)

    def forward(self, x):
        return self.module(x)

    def register_backward_hook(self):
        """
        Since we are using buckets, we are using param.main_grad instead of param.grad.
        Thus, the backward_hook that we need to register 
        """
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_hook(self._make_param_hook(param))

    def _make_param_hook(self, param):
        """
        This is the hook that is called when the gradients are available after the backward pass.
        It is called for each parameter in the model. 
        This way we can overlap computation and communication.
        """
        def param_hook(grad):
            print(f"****** Param hook called for {param.shape}. Main grad: {param.main_grad.shape}")

            # Accumulate gradients into param.main_grad.
            param.main_grad.add_(grad.data)
            # param.grad = None # clear the gradient from the param.

            # Once the accumulation is done, we need to synchronize the gradients.
            if self.require_backward_sync:
                
                # The very first time the hook is called, we need to add a hook to the autograd engine.
                # This hook is called after the backward pass through the entire model. 
                # We don't want this hook to be called at backward passes when accumulation is being done. 
                if not self._post_backward_callback_set:
                    print(f"****** Registering post backward hook. {param.name}")
                    torch.autograd.Variable._execution_engine.queue_callback(self._post_backward)
                    self._post_backward_callback_set = True
                
                # Mark the parameter as ready for synchronization.
                bucket_idx = self._param_to_bucket_location[param][2]
                self.buckets[bucket_idx].mark_param_as_ready(param)

            return torch.zeros_like(grad)

        return param_hook
    
    def _post_backward(self, *unused):
        """
        This is the callback that is called after the backward pass through the entire model.
        It is only called once after the backward pass through the entire model.
        It is hooked to the autograd engine and not to individual paramters.
        """
        for bucket in self.buckets:
            bucket.wait()
        
        self._post_backward_callback_set = False

        for bucket in self.buckets:
            bucket.reset()


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

    # for name, param in model.named_parameters():
    #     print(f"Rank {rank} param: {name} {param.shape}")

    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    model = DataParallelBucket(model, dp_process_group)

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

            optimizer.step()
            print(f"Rank {rank} param: {model.module.linear.weight[-1]}") # should be the same across ranks

    dist.destroy_process_group()


