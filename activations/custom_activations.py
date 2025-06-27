import torch.nn.functional as F

def custom_relu(x, precision='fp32'):
    if precision == 'fp8':
        x = x.half()  # Emulate with float16 for testing
    elif precision == 'fp16':
        x = x.half()
    return F.relu(x)

def custom_sigmoid(x, precision='fp32'):
    if precision == 'fp8':
        x = x.half()  # Emulate with float16 for testing
    elif precision == 'fp16':
        x = x.half()
    return F.sigmoid(x)

def custom_tanh(x, precision='fp32'):
    if precision == 'fp8':
        x = x.half()  # Emulate with float16 for testing
    elif precision == 'fp16':
        x = x.half()
    return F.tanh(x)

def custom_softmax(x, precision='fp32'):
    if precision == 'fp8':
        x = x.half()  # Emulate with float16 for testing
    elif precision == 'fp16':
        x = x.half()
    return F.softmax(x, dim=1)

def custom_leaky_relu(x, negative_slope=0.01, precision='fp32'):
    if precision == 'fp8':
        x = x.half()  # Emulate with float16 for testing
    elif precision == 'fp16':
        x = x.half()
    return F.leaky_relu(x, negative_slope=negative_slope)

def custom_swish(x, precision='fp32'):
    if precision == 'fp8':
        x = x.half()  # Emulate with float16 for testing
    elif precision == 'fp16':
        x = x.half()
    return x * F.sigmoid(x)

def custom_gelu(x, precision='fp32'):
    if precision == 'fp8':
        x = x.half()  # Emulate with float16 for testing
    elif precision == 'fp16':
        x = x.half()
    return F.gelu(x)

def custom_softplus(x, beta=1, threshold=20, precision='fp32'):
    if precision == 'fp8':
        x = x.half()  # Emulate with float16 for testing
    elif precision == 'fp16':
        x = x.half()
    return F.softplus(x, beta=beta, threshold=threshold)
