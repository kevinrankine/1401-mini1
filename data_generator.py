import numpy as np
from pprint import pprint

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
    
    return dataset

        
            
if __name__ == '__main__':
    main()
