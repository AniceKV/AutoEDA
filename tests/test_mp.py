import concurrent.futures
import time

def my_func(x):
    time.sleep(1)
    return x*x

if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(my_func, i) for i in range(5)]
        for f in concurrent.futures.as_completed(futures):
            print(f.result())
