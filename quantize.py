import torch
from torch.autograd.function import InplaceFunction, Function
import torch.nn as nn
import torch.nn.functional as F


class DirectQuant(InplaceFunction):
    """
    Quantize w.r.t. qmax in forward pass; use STE in backward pass
    """
    @staticmethod
    def forward(ctx, input, qmax, inplace=False):
        ctx.inplace = inplace
        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input

        else:
            output = input.clone()

        output = ((output * qmax).round()) / qmax

        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output
        return grad_input, None, None


class CGPACTLayer(nn.Module):
    def __init__(self, num_bits=8):
        super(CGPACTLayer, self).__init__()
        self.num_bits = num_bits
        self.qmax = 2 ** num_bits
        self.alpha = torch.nn.Parameter(torch.ones(1)*10)

    def forward(self, input):
        output = CGPACT.apply(input, self.alpha, self.qmax)

        return output


class CGPACT(Function):
    @staticmethod
    def forward(ctx, input, alpha, qmax):
        output = input.clone()
        output = (torch.abs(output) - torch.abs(output - alpha) + alpha) / 2
        qoutput = DirectQuant.apply(output/alpha, qmax)
        ctx.save_for_backward(input, alpha, qoutput)

        return alpha*qoutput

    @staticmethod
    def backward(ctx, grad_output):
        input, alpha, qoutput = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_alpha = grad_output.clone()

        grad_alpha[input < alpha] = (qoutput - input / alpha)[[input < alpha]]
        grad_alpha[input > alpha] = 1

        grad_input[input > alpha] = 0
        grad_input[input < 0] = 0

        return grad_input, grad_alpha, None


class DoReFaQuantizeLayer(nn.Module):
    """!
    Perform quantizatin of DoReFa paper on input tensor

    """
    def __init__(self, num_bits=8, quant_scale=False):
        super(DoReFaQuantizeLayer, self).__init__()
        self.num_bits = 8
        self.qmax = 2 ** num_bits
        self.quant_scale = quant_scale


    def forward(self, input):
        output = input.clone()
        # rescale to 0, 1
        output = (torch.tanh(output) / torch.max(torch.abs(torch.tanh(output.detach()))) + 1) / 2

        # quantize and dequant
        output = DirectQuant.apply(output, self.qmax)

        # scale back
        output = 2 * output - 1

        if self.quant_scale: 
            # according to original paper, this op should be replaced with modified bias term to save cimputing resource;
            # but I'll just use the same op for simplicity
            output = output / torch.sqrt(torch.var(output.detach()*output.shape[0]))

        return output


class QConv2d(nn.Conv2d):
    """docstring for QConv2d."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, num_bits=8, quant_scale=False):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)
        # self.num_bits = num_bits
        self.quant = DoReFaQuantizeLayer(num_bits=num_bits, quant_scale=quant_scale)


    def forward(self, input):
        qweight = self.quant(self.weight)
        
        output = F.conv2d(input, qweight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

        return output


class QLinear(nn.Linear):
    """docstring for QLinear."""
    def __init__(self, in_features, out_features, bias=True, num_bits=8, quant_scale=False):
        super(QLinear, self).__init__(in_features, out_features, bias)
        # self.num_bits = num_bits
        self.quant = DoReFaQuantizeLayer(num_bits=num_bits, quant_scale=quant_scale)


    def forward(self, input):
        qweight = self.quant(self.weight)

        output = F.linear(input, qweight, self.bias)

        return output