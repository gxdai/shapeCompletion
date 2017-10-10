import numpy as np
f = open('../../data/shapeCompletion/shapenet_dim32_df/02691156/fff513f407e00e85a9ced22d91ad7027__0__.df', 'rb')

for _ in range(2):
    data = f.read(8)
    print(np.fromstring(data, dtype=np.uint64))
    data = f.read(8)
    print(np.fromstring(data, dtype=np.uint64))
    data = f.read(8)
    print(np.fromstring(data, dtype=np.uint64))
    data = f.read(8)
    print(np.fromstring(data, dtype=np.float16))
