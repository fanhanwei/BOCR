import numpy as np
class FIFO:
    def __init__(self, length, num, init_method='empty'):
        self.length = length
        self.init_method = init_method
        if init_method == 'empty':
            self.mem = np.empty((num,length))
        elif init_method == 'zero':
            self.mem = np.zeros((num,length))
        elif init_method == 'None':
            self.mem = np.empty((num,length))
            for i in range(length):
                for j in range(num):
                    self.mem[i][j] = None
        else:
            raise NotImplementedError

    def _get_size(self):
        return self.mem.shape

    def _update(self, item, idx):
        self.mem = np.delete(self.mem, idx, axis = 1)
        self.mem = np.hstack((self.mem, item))

    def _items(self):
        return self.mem

if __name__ == '__main__':
    print('A test for FIFO:')
    f = FIFO(3,15,'zero')
    print(f._items())
    raitos=np.array([1,2,3,4,5,6,7,8,9,0,1,2,3,4,5])
    raitos=raitos.reshape(len(raitos),1)
    #raitos=np.array([[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2]])
    f._update(raitos,0)
    print(f._items())
    raitos=np.array([[3],[3],[3],[3],[3],[3],[3],[3],[3],[3],[3],[3],[3],[3],[3]])
    f._update(raitos,0)
    print(f._items())
    raitos=np.array([[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1]])
    f._update(raitos,0)
    print(f._items())
    print(f._items()[0])
