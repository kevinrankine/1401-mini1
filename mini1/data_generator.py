import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt

def main():
    N = 8
    K = 4
    grid = np.array([[0 for _ in range(N)] for _ in range(N)])

    dataset = {}

    # horizontal

    for i in range(N):
        for j in range(N):
            grid[i][j] = 1
        if 1 not in dataset:
            dataset[1] = [grid]
        else:
            dataset[1].append(grid)
        
        grid = np.array([[0 for _ in range(N)] for _ in range(N)])

    # vertical

    for i in range(N):
        for j in range(N):
            grid[j][i] = 1
            if 2 not in dataset:
                dataset[2] = [grid]
        else:
            dataset[2].append(grid)
        grid = np.array([[0 for _ in range(N)] for _ in range(N)])

    # ldiagonal
    
    row = 0
    col = 0

    for i in range(N):
        col = i
        row = 0

        while (col >= 0 and row < N):
            grid[row][col] = 1
            row += 1
            col -= 1
        if 3 not in dataset:
            dataset[3] = [grid]
        else:
            dataset[3].append(grid)
        grid = np.array([[0 for _ in range(N)] for _ in range(N)])

    
    for i in range(1, N):
        row = i
        col = N - 1

        while (col >= 0 and row < N):
            grid[row][col] = 1
            row += 1
            col -= 1
        if 3 not in dataset:
            dataset[3] = [grid]
        else:
            dataset[3].append(grid)
        
        grid = np.array([[0 for _ in range(N)] for _ in range(N)])

    # rdiagonal

    for i in range(N):
        col = N - 1 - i
        row = 0

        while (col < N and row < N):
            grid[row][col] = 1
            row += 1
            col += 1
        if 4 not in dataset:
            dataset[4] = [grid]
        else:
            dataset[4].append(grid)
        
        grid = np.array([[0 for _ in range(N)] for _ in range(N)])

        

    for i in range(1, N):
        row = i
        col = 0

        while (col >= 0 and row < N):
            grid[row][col] = 1
            row += 1
            col += 1
        if 4 not in dataset:
            dataset[4] = [grid]
        else:
            dataset[4].append(grid)
        
        grid = np.array([[0 for _ in range(N)] for _ in range(N)])

    dataset[5] = (dataset[2][:5] + dataset[3][4:])[1:]
    dataset[6] = dataset[2][:5] + dataset[4][::-1][4:]

    dataset[5] = (dataset[2][:5] + dataset[3][4:])[1:]
    dataset[6] = dataset[2][:5] + dataset[4][::-1][4:]
    visual_grid = np.array([[1. if 1 in dataset[1][0][i][j] else 0. for i in range(N)] for j in range(N)])
    
    plt.imshow(visual_grid, cmap='gray', interpolation='nearest')
    plt.show()
    return dataset

        
            
if __name__ == '__main__':
    main()
