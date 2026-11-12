def fs_worker(shm_name: str , slot_idx: int , iters: int) -> None:
    shm = shared_memory.SharedMemory(name=shm_name)
    arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)
    acc = arr[slot_idx]
    mask = 0xFFFFFFFFFFFFFFFF
    mult = 6364136223846793005
    for i in range(iters):
        acc = (acc * mult + (i | 1)) & mask
        arr[slot_idx] = acc # Direct hardware store to shared L1 cache line
    del arr # Release buffer export before closing SharedMemory
    shm.close()