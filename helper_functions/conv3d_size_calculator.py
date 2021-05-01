import math

# (N, Cin, D, H, W) -> (N, Cout, Dout, Hout, Wout)
N = 1 # batch size
Cin = 512 #channels in 
D = 2 # depth 
H = 45 # height
W = 80 # width

stride = (1, 1, 1)
padding = (1, 0, 0)
dilation = (1, 1, 1)
kernel_size = (2, 1, 1)

Dout = math.floor((D + 2*padding[0] - dilation[0]*(kernel_size[0]-1) -1)/stride[0] + 1)
print(Dout)
