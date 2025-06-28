import os
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# Create a simple dataset
class SimpleDataset(Dataset):
    def __init__(self, size=1000):
        self.data = torch.arange(size)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def main():
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

    # Create dataset
    dataset = SimpleDataset()
    
    # Create distributed sampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=42  # Use same seed for reproducibility
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        sampler=sampler
    )

    # Training loop simulation
    for epoch in range(2):  # Run for 2 epochs
        sampler.set_epoch(epoch)  # Important for shuffling!
        
        for batch_idx, data in enumerate(dataloader):
            # Simulate computation
            local_result = data * (rank + 1)
            
            # Gather results
            gathered_results = [torch.zeros_like(local_result) for _ in range(world_size)]
            dist.all_gather(gathered_results, local_result)
            
            full_result = torch.cat(gathered_results)
            
            print(f"Rank {rank}, Epoch {epoch}, Batch {batch_idx}")
            print(f"Local data: {data}")
            print(f"Local result: {local_result}")
            print(f"Gathered result: {full_result}")
            
            # Only process first batch for demo
            break

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
