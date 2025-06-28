import os
import argparse
from pprint import pp
import torch
import torch.nn.functional as F
import torch.distributed as dist
import time
import json

import lovely_tensors as lt; lt.monkey_patch()

from p2p_comms import pipeline_communicate, bidirectional_pipeline_communicate

# send and receive tensors from the same rank (only two ranks communicate)
# first and last rank do something different (embedding and unembedding)
# last rank computes the loss and can compute the gradient, right after the forward pass. 
# number of microbatches = grad accumulation steps
# Each microbatch is processed by all the ranks in pipeline parallelism. 

# In terms of code, each rank runs the same function. 
# 1F1B: has the same bubble as AFAB, but the memory requirements are much lower.


class Model(torch.nn.Module):
    def __init__(self, n_layers=10):
        super().__init__()
        self.n_layers = n_layers
        self.linear_layers = torch.nn.ModuleList([
            torch.nn.Linear(10, 10) for _ in range(n_layers)
        ])

        self.final_layer = torch.nn.Linear(10, 1)

    def forward(self, x):
        for layer in self.linear_layers:
            x = layer(x)
        return self.final_layer(x)
    

class PipelineParallel(torch.nn.Module):
    def __init__(self, model, pp_rank, pp_world_size):
        super().__init__()
        layer_distribution = self.distribute_layers(model.n_layers, pp_rank, pp_world_size)
        self.linear_layers = torch.nn.ModuleDict({str(i): model.linear_layers[i] for i in layer_distribution})
        self.final_layer = model.final_layer if pp_rank == pp_world_size - 1 else torch.nn.Identity()

    def distribute_layers(self, num_layers, pp_rank, pp_world_size):
        # layers_per_rank = num_layers // pp_world_size # this is even distribution
        # when the layers can't be distributed evenly, we add 1 to each available rank (starting from the first one)
        layers_per_rank = [num_layers // pp_world_size + (1 if i < num_layers % pp_world_size else 0) for i in range(pp_world_size)]
        start_layer = sum(layers_per_rank[:pp_rank])
        return range(start_layer, start_layer + layers_per_rank[pp_rank])
    
    def forward(self, input):
        x = input
        for layer in self.linear_layers.values():
            x = layer(x)    
        return self.final_layer(x)

    def backward(self, input_tensor, output_tensor, output_tensor_grad):
        # Here we make sure that input_tensor's gradient is retained after the backward pass
        # Leaves are anyhting that is created by the user - input, parameter weights, etc. 
        # Anything else is just an intermediate node. 


        # Since PP requires processing of sublayers, the input_tensor may correspond to hidden states from the previous rank
        # By default, PyTorch doesn't retain gradients for these intermediate nodes.
        # But if we want to send the gradients back to the previous ranks, we need to retain them.
        if input_tensor is not None and input_tensor.requires_grad: input_tensor.retain_grad()

        if output_tensor_grad is None:
            output_tensor_grad = torch.ones_like(output_tensor, memory_format=torch.preserve_format)

        # Finally, perform the backward pass on output_tensor, which will store gradients in input_tensor
        torch.autograd.backward(output_tensor, grad_tensors=output_tensor_grad)
        return input_tensor.grad if input_tensor is not None else None


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, size=1000):
        self.data = torch.randn(size, 10)
        self.target = F.sigmoid(torch.sum(self.data, dim=1, keepdim=True))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]


def train_pp_1f1b(model, dataloader, grad_acc_steps, pp_rank, pp_world_size, tensor_shape, event_log):
    loss = 0.0
    input_tensors, output_tensors = [], []

    # There are three phases:
    # 1. Warmup forward pass: In this phase, each rank does a forward pass and sends the output to the next rank.
    # 2. Steady state: In this phase, each rank does a forward and backward pass. It sends both to the respective ranks. 
    # 3. Cooldown backward pass: In this phase, each rank does a backward pass and sends the gradient to the previous rank. 

    num_warmup_microbatches = min(pp_world_size - pp_rank - 1, grad_acc_steps)
    num_microbatches_remaining = grad_acc_steps - num_warmup_microbatches

    print(f"[rank {pp_rank}] warmup batches: {num_warmup_microbatches} remaining batches: {num_microbatches_remaining}\n")
    print(f"[rank {pp_rank}] Warmup forward pass...\n")
    # Warmup forward pass
    for i in range(num_warmup_microbatches):
        if pp_rank == 0:
            input_tensor, _ = next(dataloader)
        else:
            event_log.append({'event': 'recv_fwd_start', 'rank': pp_rank, 'step': i, 'time': time.time()})
            input_tensor = pipeline_communicate(operation='recv_fwd', pp_rank=pp_rank, pp_world_size=pp_world_size, tensor_shape=tensor_shape)
            event_log.append({'event': 'recv_fwd_end', 'rank': pp_rank, 'step': i, 'time': time.time()})
            print(f"[rank {pp_rank}] Warmup forward pass {i} received input tensor: {input_tensor}")

        event_log.append({'event': 'fwd_start', 'rank': pp_rank, 'step': i, 'time': time.time()})
        time.sleep(PSEUDO_COMPUTATION_TIME)
        output_tensor = model.forward(input_tensor)
        event_log.append({'event': 'fwd_end', 'rank': pp_rank, 'step': i, 'time': time.time()})

        if pp_rank == pp_world_size - 1:
            event_log.append({'event': 'loss_start', 'rank': pp_rank, 'step': i, 'time': time.time()})
            _, target_tensor = next(dataloader)
            output_tensor = F.mse_loss(output_tensor, target_tensor, reduction='mean')
            event_log.append({'event': 'loss_end', 'rank': pp_rank, 'step': i, 'time': time.time()})
            loss += output_tensor.item() / grad_acc_steps 
        
        event_log.append({'event': 'send_fwd_start', 'rank': pp_rank, 'step': i, 'time': time.time()})
        sent = pipeline_communicate(operation='send_fwd', pp_rank=pp_rank, pp_world_size=pp_world_size, tensor=output_tensor)
        event_log.append({'event': 'send_fwd_end', 'rank': pp_rank, 'step': i, 'time': time.time()})
        if sent: print(f"[rank {pp_rank}] Warmup forward pass {i} Sent output tensor: {output_tensor}\n")

        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)

    # Steady State: Forward and Backward passes 
    # We want to overlap sending of fwd and bwd, so that is only possible when 
    # this is possible only after the fwd and bwd are computed
    # Thus, we start with an input_tensor from warmup phase 
    print(f"[rank {pp_rank}] Steady state...\n")
    if num_microbatches_remaining > 0:
        event_log.append({'event': 'recv_fwd_start', 'rank': pp_rank, 'step': num_warmup_microbatches, 'time': time.time()})
        input_tensor = pipeline_communicate(operation='recv_fwd', pp_rank=pp_rank, pp_world_size=pp_world_size, tensor_shape=tensor_shape)
        event_log.append({'event': 'recv_fwd_end', 'rank': pp_rank, 'step': num_warmup_microbatches, 'time': time.time()})

    for i in range(num_microbatches_remaining):
        step_idx = num_warmup_microbatches + i
        # forward step
        if input_tensor is None:
            input_tensor, _ = next(dataloader)
        event_log.append({'event': 'fwd_start', 'rank': pp_rank, 'step': step_idx, 'time': time.time()})
        time.sleep(PSEUDO_COMPUTATION_TIME)
        output_tensor = model.forward(input_tensor)
        event_log.append({'event': 'fwd_end', 'rank': pp_rank, 'step': step_idx, 'time': time.time()})

        if pp_rank == pp_world_size - 1:
            event_log.append({'event': 'loss_start', 'rank': pp_rank, 'step': step_idx, 'time': time.time()})
            _, target_tensor = next(dataloader)
            output_tensor = F.mse_loss(output_tensor, target_tensor, reduction='mean')
            event_log.append({'event': 'loss_end', 'rank': pp_rank, 'step': step_idx, 'time': time.time()})
            loss += output_tensor.item() / grad_acc_steps
        
        # send the output_tensor and recv backward, both to and from rank + 1
        event_log.append({'event': 'send_fwd_start', 'rank': pp_rank, 'step': step_idx, 'time': time.time()})
        event_log.append({'event': 'recv_bwd_start', 'rank': pp_rank, 'step': step_idx, 'time': time.time()})
        output_tensor_grad = bidirectional_pipeline_communicate(
            operation='send_fwd_recv_bwd', send_tensor=output_tensor, recv_tensor_shape=tensor_shape,
            pp_rank=pp_rank, pp_world_size=pp_world_size, 
        ) 
        event_log.append({'event': 'send_fwd_end', 'rank': pp_rank, 'step': step_idx, 'time': time.time()})
        event_log.append({'event': 'recv_bwd_end', 'rank': pp_rank, 'step': step_idx, 'time': time.time()})
        print(f"[rank {pp_rank}] Steady state pass {i} recv output tensor grad: {output_tensor_grad}")
        
        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)

        # backward step 
        input_tensor, output_tensor = input_tensors.pop(0), output_tensors.pop(0)
        event_log.append({'event': 'bwd_start', 'rank': pp_rank, 'step': i, 'time': time.time()})
        time.sleep(PSEUDO_COMPUTATION_TIME * 2)
        input_tensor_grad = model.backward(input_tensor, output_tensor, output_tensor_grad)
        event_log.append({'event': 'bwd_end', 'rank': pp_rank, 'step': i, 'time': time.time()})

        # After processing the last microbatch, we don't get any input_tensor from the previous rank. 
        if i == num_microbatches_remaining - 1:
            input_tensor = None
            event_log.append({'event': 'send_bwd_start', 'rank': pp_rank, 'step': step_idx, 'time': time.time()})
            pipeline_communicate(
                operation='send_bwd', pp_rank=pp_rank, pp_world_size=pp_world_size, tensor=input_tensor_grad
            )
            event_log.append({'event': 'send_bwd_end', 'rank': pp_rank, 'step': step_idx, 'time': time.time()})
            print(f"[rank {pp_rank}] Steady state pass {i}. Closing the steady state phase...")
        else:
            event_log.append({'event': 'send_bwd_start', 'rank': pp_rank, 'step': step_idx, 'time': time.time()})
            event_log.append({'event': 'recv_fwd_start', 'rank': pp_rank, 'step': step_idx+1, 'time': time.time()})
            input_tensor = bidirectional_pipeline_communicate(
                'send_bwd_recv_fwd', pp_rank=pp_rank, pp_world_size=pp_world_size,  
                send_tensor=input_tensor_grad, recv_tensor_shape=tensor_shape
            )
            event_log.append({'event': 'send_bwd_end', 'rank': pp_rank, 'step': step_idx, 'time': time.time()})
            event_log.append({'event': 'recv_fwd_end', 'rank': pp_rank, 'step': step_idx+1, 'time': time.time()})
            print(f"[rank {pp_rank}] Steady state pass {i} recv input tensor: {input_tensor_grad}")

    # Cooldown phase: Only send the remaining backwards 
    print(f"[rank {pp_rank}] Cooldown phase...\n")
    for i in range(num_warmup_microbatches):
        step_idx = grad_acc_steps + i
        backward_step_idx = num_microbatches_remaining + i
        input_tensor, output_tensor = input_tensors.pop(0), output_tensors.pop(0)
        
        # recv the gradient from the next rank
        event_log.append({'event': 'recv_bwd_start', 'rank': pp_rank, 'step': step_idx, 'time': time.time()})
        output_tensor_grad = pipeline_communicate(operation='recv_bwd', pp_rank=pp_rank, pp_world_size=pp_world_size, tensor_shape=tensor_shape)
        event_log.append({'event': 'recv_bwd_end', 'rank': pp_rank, 'step': step_idx, 'time': time.time()})
        
        # backward step
        event_log.append({'event': 'bwd_start', 'rank': pp_rank, 'step': backward_step_idx, 'time': time.time()})
        time.sleep(PSEUDO_COMPUTATION_TIME * 2)
        input_tensor_grad = model.backward(input_tensor, output_tensor, output_tensor_grad)
        event_log.append({'event': 'bwd_end', 'rank': pp_rank, 'step': backward_step_idx, 'time': time.time()})

        # send the gradient to the previous rank
        event_log.append({'event': 'send_bwd_start', 'rank': pp_rank, 'step': step_idx, 'time': time.time()})
        pipeline_communicate(
            'send_bwd', pp_rank=pp_rank, pp_world_size=pp_world_size, tensor=input_tensor_grad
        )
        event_log.append({'event': 'send_bwd_end', 'rank': pp_rank, 'step': step_idx, 'time': time.time()})

    return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for toy model")
    parser.add_argument("--grad_acc_steps", type=int, default=1)
    parser.add_argument("--pseudo_computation_time", type=float, default=0.0)
    args = parser.parse_args()

    PSEUDO_COMPUTATION_TIME = args.pseudo_computation_time
    grad_acc_steps = args.grad_acc_steps
    NUM_STEPS_PER_EPOCH = 1
    
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    master_addr = os.environ['MASTER_ADDR']
    master_port = os.environ['MASTER_PORT']

    print(f"[rank {rank}] grad_acc_steps: {grad_acc_steps}, world_size: {world_size}")

    dist.init_process_group(
        backend='gloo',
        init_method=f'tcp://{master_addr}:{master_port}',
        rank=rank,
        world_size=world_size
    )

    dataset = SimpleDataset()
    
    # Each rank gets the same dataloader. It is not useful for any dataloader other than rank=0. 
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=10, 
    )

    # Each backward pass is computed on grad_acc_steps * batch_size number of samples.
    
    # 1. Distribute the model across ranks
    print(f"[rank {rank}] Distributing model...")
    model = PipelineParallel(Model(), rank, world_size)
    model.train()
    dist.barrier()

    # each rank holds its own optimizer state. 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    dist.barrier()

    event_log = []
    for epoch in range(1):
        dataloader_iter = iter(dataloader)
        for batch_idx in range(NUM_STEPS_PER_EPOCH):
            acc_loss = 0.0
            print(f"[rank {rank}] Starting epoch {epoch} batch {batch_idx}")
            optimizer.zero_grad()
            loss = train_pp_1f1b(model, dataloader_iter, grad_acc_steps, rank, world_size, (10, 10), event_log)
            acc_loss += loss
            
            event_log.append({'event': 'optimizer_start', 'rank': rank, 'epoch': epoch, 'batch': batch_idx, 'time': time.time()})
            time.sleep(PSEUDO_COMPUTATION_TIME)
            optimizer.step()
            event_log.append({'event': 'optimizer_end', 'rank': rank, 'epoch': epoch, 'batch': batch_idx, 'time': time.time()})
            
            print(f"[rank {rank}] Finished epoch {epoch} batch {batch_idx} with loss {acc_loss}")
        
        # Save event log
        os.makedirs("events_1f1b", exist_ok=True)
        with open(f"events_1f1b/event_log_rank{rank}.json", "w") as f:
            json.dump(event_log, f)
    dist.destroy_process_group()


