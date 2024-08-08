import numpy as np
 
def output_func(act):
    if act >= 0:
        return 1
    return 0


input_list = np.array([
[1, 1, 0],
[1, 2, 0],
[1, 3, 0],
[1, 0, 1],
[1, 0, 2]
])
 
output_list = np.array([
0,
0,
1,
0,
1
])
 
weights = np.array([3, 1, 2])
 
weights_update = [np.zeros(len(output_list)) ]
 
 
cnt = 5
epoch = 1
 
while cnt > 0:
    print(f"Epoch {epoch}")
    for i in range(len(input_list)):
        act = np.sum(input_list[i] * weights)
        error = output_list[i] - output_func(act)
        weights_update = error * input_list[i]
        if np.any(weights_update):
            cnt = 5
            weights = weights + weights_update
        cnt -= 1
    epoch += 1
    print(weights)
    print()
 
print(f"final weights {weights}")