# The Parity Check

`mbarrier.arrive` gives you a token representing the past phase (the return value). When your thread arrives at the barrier, the barrier is in a certain phase (let's say Phase N). The arrive instruction gives you a 64-bit token that is a handle or a receipt for "Phase N".

`mbarrier.wait.parity` uses that token to wait for the future phase. The wait instruction doesn't need to know the exact number of the next phase (N+1). It only needs to know that the phase has changed. The simplest way to check for a change is to check if the phase number has flipped from even to odd, or from odd to even. This is the parity check.

The `wait` operation uses the token you give it to ask the hardware a simple question:

> "Does the barrier's current phase have a different parity than the phase encoded in my token?"

If the answer is "yes," the wait succeeds. If "no," the thread keeps waiting.

But in the [implementation](https://github.com/lcy-seso/DLFrameworkTest/blob/master/Cuda/hopper_gemm/tma_load_cuda/tma_utils.cuh#L219), ptx instruction used in the `arrive` operation uses the sink symbol `_`:

```cpp
asm volatile(
    "{\n\t"
    "mbarrier.arrive.shared::cta.b64 _, [%0];\n\t"  // "_" sink symbol is used
    "}"
    :
    : "r"(barrier_ptr));
```

It means that the `arrive` operation is not retrieving a phase token.

Additionally, in the [`wait` operation](https://github.com/lcy-seso/DLFrameworkTest/blob/master/Cuda/hopper_gemm/tma_load_cuda/tma_utils.cuh#L198), we can see that: instead of passing a 64-bit state token, an immediate integer: 0 (for even) or 1 (for odd) is passed to tell the hardware the parity value directly.

Note that the thread calling `arrive` with `_` is not the same thread that will be calling `wait`. This is a classic Producer-Consumer pattern, also known as a **Signaler-Waiter** pattern.

Let's break down the roles:

1. **The Signaler (Producer)**

    - **Job**: To perform a task (like copying data from global to shared memory) and then signal to other threads that the task is complete.
    - **Action**: It calls `mbarrier.arrive.shared::cta.b64 _, [%barrier_addr];`.
    
    <ins>***Why it uses the sink symbol `_` ?***</ins>
    
    The signaler's job is done after it raises the flag. It doesn't need to wait for anyone else. It might go on to do other work or simply terminate. It has no use for the phase token, so it discards it for efficiency. It is purely producing a signal.

    The TMA units are hardware producers; their software-controlled part simply needs to arrive at a barrier to indicate "the data you asked for is now in shared memory."

2. **The Waiter (Consumer)**

    - **Job**: To wait until a certain condition is met (e.g., data is ready in shared memory) before it can start its own work (like performing matrix multiplications).
    - **Action**: It calls `mbarrier.try_wait.parity ... [%barrier_addr], %phase_token;`.
    
    <ins>***Where its token comes from ?***</ins>
    
    This is the key. The consumer's phase token does not come from the signaler's arrive call. The consumer must have acquired its token from a previous synchronization event in its own logical flow.

### A Concrete Example

Imagine a pipelined GEMM kernel. You have two sets of threads (or thread roles):

- Copy Threads (Producers/Signalers): Use TMA to copy tile A of a matrix into shared memory.
- Math Threads (Consumers/Waiters): Use the data in shared memory to perform MMA (Matrix-Multiply-Accumulate) instructions.

You'll use two barriers to manage the shared memory buffer:

- `full_barrier`: Signals that the buffer is full of valid data.
- `empty_barrier`: Signals that the buffer has been consumed and is empty.

Here's the dance, focusing on one buffer:

#### Initial State

- `full_barrier` is at Phase 0 (even).
- `empty_barrier` is at Phase 0 (even).
- The Math Thread has a token for `full_barrier`'s Phase 0.
- The Copy Thread has a token for `empty_barrier`'s Phase 0.

#### Cycle 1

- **COPY THREAD (Producer):**

  - Waits on `empty_barrier` using its "Phase 0" token. This succeeds immediately because the buffer starts empty.
  - Copies data tile A into the shared memory buffer.
  - Signals completion by calling arrive on `full_barrier`. Since it's just signaling, it uses `_`: `full_barrier.arrive(_, ...);`.
  - When all copy threads are done, `full_barrier`'s phase flips to 1 (odd).

- **MATH THREAD (Consumer):**

  - Waits on `full_barrier` using its "Phase 0" token. It spins, repeatedly calling `try_wait`.
  - The check `current_parity(odd) != token_parity(even)` is now true. The wait succeeds!
  - The Math Thread consumes the data (performs MMA).
  - After consuming, it signals that the buffer is now empty by calling arrive on `empty_barrier`. It's just signaling, so it uses `_`: `empty_barrier.arrive(_, ...);`.
  - When all math threads are done, `empty_barrier`'s phase flips to 1 (odd).

The cycle can now repeat for the next data tile. The Copy Thread will wait for `empty_barrier`'s phase to become 1, and the Math Thread will wait for `full_barrier`'s phase to become 1.

### Conclusion

The pattern in use is the core of how asynchronous operations are synchronized within a CUDA block. The `arrive` is a "fire-and-forget" signal from a producer. The `try_wait` is a consumer polling for that signal, using a token it acquired from a different, previous point in its own execution timeline to know which signal phase it's waiting for.
