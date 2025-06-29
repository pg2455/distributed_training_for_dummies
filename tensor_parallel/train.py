import os
import argparse
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import math

def print_if_rank_0(rank, message):
    if rank == 0:
        print(message)

## TP Comunication Primitives
# Here we define the forward-backward of distributed operations. 
# Note: We can use functions from torch.distributed but here we define them from scratch for better understanding.

# We subclass autograd function
class Copy(torch.autograd.Function):
    """It copies the input tensor to the rank. 
    While the forward pass doesn't do anything, the backward pass needs to aggregate gradients. 
    Thus, Forward is a no-op and backward is an all-reduce. 

    Forward: broadcast
    Backward: all_reduce
    """
    @staticmethod 
    def forward(ctx, input):
        ## in ctx we can store any infomration that we want to use in the backward pass. 
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # The gradients need to be summed & synced across all ranks since the inputs are copied to all ranks.
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM, group=None)
        return grad_output


class Gather(torch.autograd.Function):
    """When the input is sharded across the ranks along a dimension,
    Gather gathers the sharded input and concatenates them along the dimension.
    We assume that the last dim is the dimension along which the input is sharded.

    Forward: Concat across the ranks 
    Backward: Sum the gradients and scatter them (reduce-scatter)

    """
    @staticmethod
    def forward(ctx, input, tp_rank):
        ctx.tp_rank = tp_rank 
        ctx.tp_world_size = tp_world_size = dist.get_world_size() 
        if tp_world_size == 1:
            return input 

        last_dim = input.dim() - 1
        input = input.contiguous()
        tensor_list = [torch.empty_like(input) for _ in range(tp_world_size)]

        tensor_list[tp_rank] = input
        dist.all_gather(tensor_list, input)
        output = torch.cat(tensor_list, dim=last_dim).contiguous()
        return output
    
    @staticmethod 
    def backward(ctx, grad_output):
        if ctx.tp_world_size == 1:
            return grad_output, None 
        
        # Split gradients so that each process get the gradients corresponding to its part of the tensor 
        last_dim = grad_output.dim() - 1
        assert grad_output.size()[last_dim] % ctx.tp_world_size == 0
        last_dim_size = grad_output.size()[last_dim] // ctx.tp_world_size 
        chunks = torch.split(grad_output, last_dim_size, dim=last_dim)
        return chunks[ctx.tp_rank].contiguous(), None


class Reduce(torch.autograd.Function):
    """
    All reduce

    If the output of all_reduce is used in the loss function directly, it is reduce and and not all reduce. Backward of reduce will be identity.
    If the output of all_reduce is used differently across each rank (e.g., use another ColumnParallel linear layer), we need to sum the gradients and sync them.

    Forward: Reduce -> Copy 
    Backward: Reduce -> Copy 
    """
    @staticmethod
    def forward(ctx, input, tp_rank):
        ctx.tp_rank = tp_rank 
        ctx.tp_world_size = dist.get_world_size()
        if ctx.tp_world_size == 1:
            return input # nothing to reduce
        
        # All reduce sums and syncs the result, so after this line, input will be same across the ranks
        dist.all_reduce(input, op=dist.ReduceOp.SUM, group=None)
        return input
    
    @staticmethod
    def backward(ctx, grad_output):
        if ctx.tp_world_size == 1:
            return grad_output, None
        
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM, group=None)
        return grad_output, None

## TP Communication ends

# Notation: Y = X * W + b, X = [B, D], W = [D, H], b = [H]
# This class shards weight parameters column wise, i.e., On each rank, we have H // tp_world_size columns of W. 
# The output consists of H // tp_world_size columns of Y. 
class ColumnParallelLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, tp_rank, gather_outputs=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tp_rank = tp_rank
        self.tp_world_size = dist.get_world_size()
        self.gather_outputs = gather_outputs

        # we shard the weight parameters column wise, i.e., input_features // tp_world_size parameters
        # remain on each rank. 
        assert out_features % self.tp_world_size == 0, f"out_features: {out_features} should be divisible by tp_world_size: {self.tp_world_size}"
        self.out_features_per_rank = out_features // self.tp_world_size
        self.weight = torch.nn.Parameter(torch.Tensor(self.out_features_per_rank, in_features))
        self.bias = torch.nn.Parameter(torch.zeros(self.out_features_per_rank))
        self.reset_parameters()

    def reset_parameters(self):
        """
        This function is responsible for sharding the weight parameters across ranks. 
        Each rank calls this function and gets corresponding weight parameters. 
        """
        # Initialize weight tensor with the default initialization method used for nn.Linear in PyTorch
        if self.tp_world_size == 1:
            #  U(-sqrt(k), sqrt(k))
            k = 1 / self.weight.size(1)
            bound = math.sqrt(k)
            torch.nn.init.uniform_(self.weight, -bound, bound)
            return
        
        # When TP > 1, Initialize master weight
        master_weight = torch.empty(self.out_features, self.in_features, dtype=self.weight.dtype, requires_grad=False)
        k = 1 / master_weight.size(1)
        bound = math.sqrt(k)
        torch.nn.init.uniform_(master_weight, -bound, bound)

        # Sharding: Now we split the master weights into tp_world_size groups and store the one corresponding to the current rank. 
        # Note: some pytorch operations like split, can return non-contiguous tensors, i.e., the parameters are not contained in a continuous block of memory.
        # Since most operations like matmul require contiguous tensors, we need to make sure that the weight tensor after splitting is contiguous.
        weight_list = torch.split(master_weight, self.out_features_per_rank, dim=0)
        self.weight.data = weight_list[self.tp_rank].contiguous()

    def forward(self, x):
        # Each rank gets a copy of the input tensor. 
        input_broadcasted = Copy.apply(x)
        output = F.linear(input_broadcasted, self.weight, self.bias)
        if self.gather_outputs:
            output = Gather.apply(output, self.tp_rank)
        return output


# Here we shard weights along the input_features, so that W is sharded across ranks 
# Such that each rank gets D // tp_world_size number of rows. 
# The output will have all the columns except that it will be partially computed using the partial rows on each rank.
class RowParallelLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, tp_rank):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.tp_rank = tp_rank
        self.tp_world_size = dist.get_world_size()
        assert in_features % self.tp_world_size == 0, "Hidden dimension must be divisible by the tensor parallel world size"
        self.input_size_per_rank = in_features // self.tp_world_size

        self.weight = torch.nn.Parameter(torch.randn(out_features, self.input_size_per_rank))
        self.bias = torch.nn.Parameter(torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        """
        We shard the weights here.
        """
        if self.tp_world_size == 1:
            k = 1 / self.weight.size()[1] # input_dimension
            bound = math.sqrt(k)
            torch.nn.init.uniform_(self.weight, -bound, bound)
            return 
        
        # When TP group > 1, we shard the weights along the rows 
        k = 1 / self.weight.size()[1]
        bound = math.sqrt(k)
        master_weight = torch.empty(self.out_features, self.in_features, dtype=self.weight.dtype, requires_grad=False)
        torch.nn.init.uniform_(master_weight, -bound, bound)

        weight_list = torch.split(master_weight, self.input_size_per_rank, dim=1)
        self.weight.data = weight_list[self.tp_rank].contiguous()


    def forward(self, input):
        # this input is assumed to be already sharred along the feature dimension
        # Note: bias is added only once at the end once we all reduce the output
        output_parallel = F.linear(input, self.weight)
        output = Reduce.apply(output_parallel, self.tp_rank)
        return output + self.bias 


class Model(torch.nn.Module):
    def __init__(self, tp_rank):
        super().__init__()
        self.linear = ColumnParallelLinear(10, 10, tp_rank)
        self.activation = torch.nn.ReLU()
        self.linear2 = RowParallelLinear(10, 10, tp_rank) # output will be same across the ranks
        # final output will be gathered across the ranks to result in 4-dimensional vector
        self.linear3 = ColumnParallelLinear(10, 4, tp_rank, gather_outputs=True)

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        return x

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, size=1000):
        self.data = torch.randn(size, 10)
        self.target = torch.randint(0, 4, (size, 1)) # one of 4 labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]


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

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")

    print_if_rank_0(rank, f"grad_acc_steps: {grad_acc_steps}")
    print_if_rank_0(rank, f"world_size: {world_size}")

    # Initialize the process group for the tensor parallel group.
    # After this, any call to dist will assume the default process group. group=None is the default process group.
    dist.init_process_group(
        backend='gloo',
        init_method=f'tcp://{master_addr}:{master_port}',
        rank=rank,
        world_size=world_size
    )

    dataset = SimpleDataset()
    
    # This dataloader is not distributed, i.e., each tp group gets the same data. 
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=10, 
    )
    
    # 1. Share the same model across ranks.
    # each rank has its own copy of the same model weights.
    model = Model(tp_rank=rank) 
    # Once the model is intialized above, its already sharded on CPU
    # Now we move the respective shards to the GPU.
    model.to(device) 

    # 1.b Another way to have the same model across ranks is to set the same seed for each rank.

    # each rank holds its own portion of the optimizer state.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(2):
        dataloader_iter = iter(dataloader)
        
        for batch_idx in range(NUM_STEPS_PER_EPOCH):
            acc_loss = 0.0
            optimizer.zero_grad()
            for i in range(grad_acc_steps):
                batch = next(dataloader_iter)
                data, target = batch # each rank gets a unique, non-overlapping subset of data. 

                print(f"Rank {rank} data: {data.shape} target: {target.shape}")
                result = model(data)
                
                # Use cross entropy loss since target is an integer label
                loss = F.cross_entropy(result, target.squeeze(-1))
                loss.backward() 
                acc_loss += loss.item()
                print(f"Rank {rank}, Epoch {epoch}, Batch {batch_idx}, Microstep: {i}, Local loss: {loss}")

            print(f"Rank {rank}, Epoch {epoch}, Batch {batch_idx}, Accumulated loss: {acc_loss}")
            dist.barrier() # wait for all ranks to finish the current batch

            optimizer.step()
            print(f"Rank {rank} param: {model.linear.weight[-1]}") # should be the same across ranks

    dist.destroy_process_group()


