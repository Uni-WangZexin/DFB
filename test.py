import torch
a = torch.tensor([[2],[3],[4],[5]]).reshape(-1)
b = torch.tensor([[1],[0],[1],[0]]).reshape(-1)
c = torch.nonzero(b<1,as_tuple = False)
d = torch.index_select
print(a)
print(c)
print(a[c].reshape(-1,1))
print('ss')
print('ss')