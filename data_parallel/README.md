## Data Parallelism (DP) for toy model

Data Parallelism requires replicating the model across GPUs.
Each GPU handles a **microbatch** by performing a forward pass and computing its gradients in the backward pass.
These gradients are synced across all the devices using (`all_reduce`), and the optimizer performs the gradient step after that.
The order of gradient computation and communication gives rise to several possibilities.

When we divide a batch across different processes, we call each batch handled by an individual process a **microbatch**.

### Gradient Accumulation
When the batch size is too large to fit in memory, accumulating gradients across minibatches helps.
For each minibatch, a forward and backward pass is performed, accumulating gradients for the respective parameters.

### Naive DP
In its plain form, we can sync the gradients once they have been accumulated for each minibatch on each process.
However, this is inefficient because the GPUs have to wait for all gradients before taking the gradient step.

Both `train_naive_grad_accumulation.py` and `train_naive.py` implement this strategy.

### Overlap Communication with Computation

One way to reduce communication overhead is to overlap the communication of gradients with computation.
Since the communication takes place asynchronously, the processor can continue computing operations while the gradients are being passed in the background.
A way to accomplish this is by `all_reduce`-ing the gradients as soon as they are computed for each minibatch on the processor.

`train_overlap.py` implements this approach.


### Bucket Overlap
Too much communication can hinder computation.
This can be prevented by dividing the parameters into buckets, and when their gradients are ready, initiating communication to sync these gradients.

`train_bucket_overlap.py` implements this approach.