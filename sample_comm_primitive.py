import os
import torch
import torch.distributed as dist
import random

def main():
    # use of communication primitive in distributed computing
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    master_addr = os.environ['MASTER_ADDR']
    master_port = os.environ.get('MASTER_PORT', '12355')

    dist.init_process_group(
        backend='gloo',
        init_method=f'tcp://{master_addr}:{master_port}',
        rank=rank,
        world_size=world_size
    )

    # Only rank 0 creates the initial data
    if rank == 0:
        full_data = torch.arange(4) + random.randint(0, 100)
    else:
        full_data = torch.zeros(4, dtype=torch.long)
    
    # Broadcast full_data from rank 0 to all workers
    dist.broadcast(full_data, src=0) # p2p communication
    
    # Now split the data for each worker
    chunk_size = full_data.size(0) // world_size
    start = rank * chunk_size
    end = start + chunk_size
    local_data = full_data[start:end].float()

    print(f"Rank {rank} full data: {full_data}")
    print(f"Rank {rank} received data: {local_data}")

    # Each worker does some computation
    local_result = local_data * (rank + 1)
    print(f"Rank {rank} local result: {local_result}")

    # Make sure local_result is the same size on all workers
    print(f"Rank {rank} local_result shape: {local_result.shape}")
    
    # Create list of tensors for gathering results
    gathered_results = [torch.zeros_like(local_result) for _ in range(world_size)]
    
    # Gather results from all workers
    dist.all_gather(gathered_results, local_result) # collective communication
    
    # Concatenate results
    full_result = torch.cat(gathered_results)
    print(f"Rank {rank} final gathered result: {full_result}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
