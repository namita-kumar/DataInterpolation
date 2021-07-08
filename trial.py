from multiprocessing import Pool, RawArray
import pdb
from functools import partial
import numpy as np

def sum_up_to(arg):
    number, name, location = arg
    print(number, name, location)
    return sum(number)


if __name__ == "__main__":
    name = "nami"
    tasks = (([1, 2, 3, 4], name, "florida"), ([4,5,6,7], name, "china"), ([8,9,10,11], name, "india"))
    data = np.array([1,2,3],[4,5,6])
    X = RawArray('d',2 * 3)
    X_np = np.frombuffer(X,dtype=np.float64).reshape((2,3))
    np.copyto(X_np,data)
    tasks_gen = (x for x in tasks)
    del tasks
    result = []
    pdb.set_trace()

    # number = multiprocessing.Value('d', 0.0)
    # x = multiprocessing.Array('d', x)
    # p = multiprocessing.Process(target=sum_up_to, args=(number, x))
    # p.start()
    # p.join()
    # a_pool = multiprocessing.Pool(initializer=sum_up_to,initargs=(x,))
    a_pool = Pool()
    result = a_pool.map(sum_up_to, tasks)
    print(result)
