# simple_shared_test.py
import numpy as np
from multiprocessing import Process, shared_memory, Lock
import time
import pickle
import struct
import psutil

class SimpleSharedDict:
    """Super simple shared dictionary for testing"""
    
    def __init__(self, create=False, name=None, size=1024*1024):  # 1MB
        if create:
            self.shm = shared_memory.SharedMemory(create=True, size=size)
            self.name = self.shm.name
            # Initialize with empty dict
            empty_dict = {}
            self._write_dict(empty_dict)
        else:
            self.shm = shared_memory.SharedMemory(name=name)
            self.name = name
    
    def _write_dict(self, data_dict):
        """Write dictionary to shared memory"""
        data_bytes = pickle.dumps(data_dict)
        data_size = len(data_bytes)
        
        # Write size first (8 bytes), then data
        struct.pack_into('Q', self.shm.buf, 0, data_size)
        self.shm.buf[8:8+data_size] = data_bytes
    
    def _read_dict(self):
        """Read dictionary from shared memory"""
        data_size = struct.unpack_from('Q', self.shm.buf, 0)[0]
        if data_size == 0:
            return {}
        data_bytes = self.shm.buf[8:8+data_size]
        return pickle.loads(data_bytes)
    
    def get(self, key):
        """Get value by key"""
        data = self._read_dict()
        return data.get(key)
    
    def put(self, key, value):
        """Put key-value pair"""
        data = self._read_dict()
        data[key] = value
        self._write_dict(data)
    
    def size(self):
        """Get number of items"""
        data = self._read_dict()
        return len(data)
    
    def close(self):
        self.shm.close()
    
    def unlink(self):
        self.shm.unlink()

def simple_worker(cache_name, lock, worker_id):
    """Simple worker that just stores and retrieves values"""
    print(f"Worker {worker_id}: Starting")
    
    # Connect to shared cache
    cache = SimpleSharedDict(name=cache_name)
    
    # Do some work
    for i in range(3):
        key = f"worker_{worker_id}_item_{i}"
        value = worker_id * 100 + i
        
        with lock:
            # Store something
            cache.put(key, value)
            print(f"Worker {worker_id}: Stored {key} = {value}")
            
            # Try to read what another worker stored
            other_key = f"worker_{(worker_id+1)%3}_item_0"
            other_value = cache.get(other_key)
            if other_value is not None:
                print(f"Worker {worker_id}: Read {other_key} = {other_value}")
            
            print(f"Worker {worker_id}: Cache size is now {cache.size()}")
        
        time.sleep(0.5)
    
    cache.close()
    print(f"Worker {worker_id}: Finished")

def cpu_intensive_worker(cache_name, lock, worker_id):
    """Worker that actually uses CPU"""
    cache = SimpleSharedDict(name=cache_name)
    
    for i in range(3):
        # CPU-intensive work instead of sleep
        result = 0
        for j in range(10_000_000):  # Count to 10 million
            result += j * worker_id
        
        key = f"worker_{worker_id}_result_{i}"
        value = result % 1000
        
        with lock:
            cache.put(key, value)
            print(f"Worker {worker_id}: Computed {key} = {value}")
    
    cache.close()

def single_process_version():
    """Single process version doing the same work"""
    print("=== Single Process Version ===")
    
    start_time = time.time()
    
    # Create a regular dictionary (no shared memory needed)
    cache = {}
    
    # Do the same work as 3 workers, but sequentially
    for worker_id in range(5):
        print(f"Worker {worker_id}: Starting")
        
        for i in range(3):
            key = f"worker_{worker_id}_item_{i}"
            value = worker_id * 100 + i
            
            # Store something
            cache[key] = value
            print(f"Worker {worker_id}: Stored {key} = {value}")
            
            # Try to read what "another worker" stored
            other_key = f"worker_{(worker_id+1)%3}_item_0"
            other_value = cache.get(other_key)
            if other_value is not None:
                print(f"Worker {worker_id}: Read {other_key} = {other_value}")
            
            print(f"Worker {worker_id}: Cache size is now {len(cache)}")
            
            time.sleep(0.5)  # Same delay
        
        print(f"Worker {worker_id}: Finished")
    
    end_time = time.time()
    
    print(f"\nFinal cache contents: {cache}")
    print(f"Total items: {len(cache)}")
    print(f"Single process time: {end_time - start_time:.2f} seconds")
    
    return end_time - start_time

def multiprocess_version():
    """Multiprocess version (your existing code) with timing"""
    print("=== Multiprocess Shared Memory Version ===")
    
    start_time = time.time()
    
    # Create shared cache
    cache = SimpleSharedDict(create=True)
    lock = Lock()
    
    print(f"Created shared cache: {cache.name}")
    
    # Start 3 workers
    processes = []
    for i in range(9):
        p = Process(target=simple_worker, args=(cache.name, lock, i))
        processes.append(p)
        p.start()
    
    # Wait for all workers
    for p in processes:
        p.join()
    
    # Check final state
    print("\nFinal Results:")
    final_data = cache._read_dict()
    print(f"Final cache contents: {final_data}")
    print(f"Total items: {len(final_data)}")
    
    end_time = time.time()
    print(f"Multiprocess time: {end_time - start_time:.2f} seconds")
    
    # Clean up
    cache.close()
    cache.unlink()
    
    return end_time - start_time

def stress_test_processes():
    """CAREFUL: This tests system limits"""
    
    print("=== STRESS TEST WARNING ===")
    print("This will create many processes and may slow your system!")
    response = input("Continue? (type 'yes'): ")
    
    if response.lower() != 'yes':
        print("Cancelled")
        return
    
    max_processes = 50  # Start conservatively
    
    for num_proc in [5, 10, 15, 20, 25, 30]:
        print(f"\nTesting {num_proc} processes...")
        
        # Monitor system before
        cpu_before = psutil.cpu_percent()
        memory_before = psutil.virtual_memory().percent
        
        try:
            start_time = time.time()
            
            # Create shared cache
            cache = SimpleSharedDict(create=True)
            lock = Lock()
            
            # Start processes
            processes = []
            for i in range(num_proc):
                p = Process(target=simple_worker, args=(cache.name, lock, i))
                processes.append(p)
                p.start()
            
            # Monitor during
            time.sleep(1)
            cpu_during = psutil.cpu_percent()
            memory_during = psutil.virtual_memory().percent
            
            # Wait for completion
            for p in processes:
                p.join(timeout=30)  # 30 second timeout
                if p.is_alive():
                    print(f"Process {p.pid} timed out, terminating...")
                    p.terminate()
                    p.join()
            
            end_time = time.time()
            
            print(f"  Time: {end_time - start_time:.2f}s")
            print(f"  CPU: {cpu_before:.1f}% -> {cpu_during:.1f}%")
            print(f"  Memory: {memory_before:.1f}% -> {memory_during:.1f}%")
            
            cache.close()
            cache.unlink()
            
            # Stop if system is getting stressed
            if memory_during > 85 or cpu_during > 95:
                print("System getting stressed, stopping test")
                break
                
        except Exception as e:
            print(f"Error with {num_proc} processes: {e}")
            break
    
    print("\nStress test complete")

def cpu_stress_test():
    """This WILL stress your 4 cores"""
    print("=== CPU Stress Test ===")
    print("This will actually use your CPU cores!")
    
    cache = SimpleSharedDict(create=True)
    lock = Lock()
    
    # Try different numbers of CPU-intensive processes
    for num_proc in [2, 4, 8, 12]:
        print(f"\nTesting {num_proc} CPU-intensive processes...")
        start_time = time.time()
        
        processes = []
        for i in range(num_proc):
            p = Process(target=cpu_intensive_worker, args=(cache.name, lock, i))
            processes.append(p)
            p.start()
        
        # Monitor CPU usage
        time.sleep(1)
        cpu_usage = psutil.cpu_percent(interval=1)
        print(f"CPU usage with {num_proc} processes: {cpu_usage:.1f}%")
        
        for p in processes:
            p.join()
        
        end_time = time.time()
        print(f"Time for {num_proc} processes: {end_time - start_time:.2f}s")
    
    cache.close()
    cache.unlink()

if __name__ == '__main__':
    # stress_test_processes()
    cpu_stress_test()
    # # Run both versions and compare
    # print("Running performance comparison...\n")
    
    # # Test multiprocess version
    # multiprocess_time = multiprocess_version()
    
    # print("\n" + "="*50 + "\n")
    
    # # Test single process version
    # single_process_time = single_process_version()
    
    # print("\n" + "="*50)
    # print("=== PERFORMANCE COMPARISON ===")
    # print(f"Single process time:   {single_process_time:.2f} seconds")
    # print(f"Multiprocess time:     {multiprocess_time:.2f} seconds")
    
    # if multiprocess_time < single_process_time:
    #     speedup = single_process_time / multiprocess_time
    #     print(f"Multiprocess is {speedup:.2f}x FASTER")
    # else:
    #     slowdown = multiprocess_time / single_process_time
    #     print(f"Multiprocess is {slowdown:.2f}x SLOWER")
    
    # print(f"Difference: {abs(multiprocess_time - single_process_time):.2f} seconds")