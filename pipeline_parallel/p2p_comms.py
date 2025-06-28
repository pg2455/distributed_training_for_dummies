import torch
import torch.distributed as dist


def pipeline_communicate(operation, pp_rank, pp_world_size, tensor=None, tensor_shape=None):
    print(f"[rank {pp_rank}] Pipeline communicate: {operation}")
    if operation == "send_fwd":
        dest = pp_rank + 1 if pp_rank != pp_world_size - 1 else None
    
    elif operation == "recv_fwd":
        src = pp_rank - 1 if pp_rank != 0 else None
        tensor = torch.empty(tensor_shape, requires_grad=True)
    
    elif operation == "send_bwd":
        dest = pp_rank - 1 if pp_rank != 0 else None 
    
    elif operation == "recv_bwd":
        src = pp_rank + 1 if pp_rank != pp_world_size - 1 else None
        tensor = torch.empty(tensor_shape, requires_grad=True)
    

    is_send = operation.startswith('send')
    peer_rank = dest if is_send else src
    if peer_rank is None:
        print(f"[rank {pp_rank}] Peer rank is None")
        return None

    op = dist.P2POp(op=dist.isend if is_send else dist.irecv, tensor=tensor, peer=peer_rank)

    [req.wait() for req in dist.batch_isend_irecv([op])]
    return None if is_send else tensor


def bidirectional_pipeline_communicate(operation, pp_rank, pp_world_size, send_tensor=None, recv_tensor_shape=None):
    assert operation in ['send_fwd_recv_bwd', 'send_bwd_recv_fwd']

    print(f"[rank {pp_rank}] Bidirectional pipeline communicate: {operation}")
    
    # skip if its terminal ranks
    if (
        (operation == 'send_bwd_recv_fwd' and pp_rank == 0) or 
        (operation == 'send_fwd_recv_bwd' and pp_rank == pp_world_size - 1)
    ):
        return None 
    
    # fwd is sent to rank+1, fwd is recieved from rank - 1
    # bwd is sent to rank - 1, bwd is received from rank + 1

    if operation == 'send_bwd_recv_fwd':
        src, dest = pp_rank - 1, pp_rank - 1
    else:
        src, dest = pp_rank + 1, pp_rank + 1

    recv_tensor = torch.empty(recv_tensor_shape, requires_grad=True)


    reqs = dist.batch_isend_irecv([
            dist.P2POp(dist.isend, send_tensor, peer=dest),
            dist.P2POp(dist.irecv, recv_tensor, peer=src)   
    ])

    [req.wait() for req in reqs]
    return recv_tensor

    
