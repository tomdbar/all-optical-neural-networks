from abc import abstractmethod
from enum import Enum

import matplotlib.pyplot as plt
import torch

try:
    import seaborn as sns
    plt.style.use('seaborn')
except ImportError:
    pass

import numpy as np

class Encoding(Enum):
    INTENSITY = 1
    AMPLITUDE = 2

class Gradient(Enum):
    EXACT = 1
    APPROXIMATE = 2
    ZERO = 3
    POSITIVE = 4
    NEGATIVE = 5

class SatAbsNL(torch.nn.Module):

    def __init__(self, encoding=Encoding.INTENSITY, gradient=Gradient.APPROXIMATE, OD=5, I_sat=1):
        super().__init__()

        self.sat_abs_nl_func = self.__get_sat_abs_nl_func(encoding, gradient).apply
        self.encoding = encoding
        self.gradient = gradient
        self.OD = OD
        self.I_sat = I_sat

    def forward(self, input):
        return self.sat_abs_nl_func(input, self.OD, self.I_sat)

    def __get_sat_abs_nl_func(self, encoding, gradient):
        sat_abs_nl_func = None

        if encoding == Encoding.INTENSITY:
            if gradient == gradient.APPROXIMATE:
                sat_abs_nl_func = SatAbsNL_I_approxGrad
            elif gradient == gradient.EXACT:
                sat_abs_nl_func = SatAbsNL_I_exactGrad
            elif gradient == gradient.ZERO:
                sat_abs_nl_func = SatAbsNL_I_zeroGrad
            elif gradient == gradient.POSITIVE:
                sat_abs_nl_func = SatAbsNL_I_positiveGrad
            elif gradient == gradient.NEGATIVE:
                sat_abs_nl_func = SatAbsNL_I_negativeGrad

        elif encoding == Encoding.AMPLITUDE:
            if gradient == gradient.APPROXIMATE:
                sat_abs_nl_func = SatAbsNL_E_approxGrad
            elif gradient == gradient.EXACT:
                sat_abs_nl_func = SatAbsNL_E_exactGrad
            elif gradient == gradient.ZERO:
                sat_abs_nl_func = SatAbsNL_E_zeroGrad
            elif gradient == gradient.POSITIVE:
                sat_abs_nl_func = SatAbsNL_E_positiveGrad
            elif gradient == gradient.NEGATIVE:
                sat_abs_nl_func = SatAbsNL_E_negativeGrad

        if sat_abs_nl_func is None:
            print("Unrecognised options for saturated absorption non-linearity:\n\tencoding={}\n\tgradient={}".format(
                encoding, gradient
            ))

        return sat_abs_nl_func

    def extra_repr(self):
        return 'encoding={}, gradient={}, OD={}, I_sat={}'.format(
            self.encoding, self.gradient, self.OD, self.I_sat
        )

class SatAbsNL_I(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, I_in, OD, I_sat):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(I_in)
        ctx.OD = OD
        ctx.I_sat = I_sat

        return I_in * torch.exp(-OD / (1 + I_in/I_sat))

    @staticmethod
    @abstractmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        raise NotImplementedError()

class SatAbsNL_I_exactGrad(SatAbsNL_I):
    @staticmethod
    def backward(ctx, grad_output):
        I_in, = ctx.saved_tensors
        OD, I_sat = ctx.OD, ctx.I_sat
        # Return gradients.  Note as forward takes 3 arguments, we must return 3
        # gradients, however the "gradient"'s of OD and I_sat are None.
        return grad_output * (1 + (I_in/I_sat)*( OD/ (1+I_in/I_sat)**2 )) * torch.exp(-OD / (1 + I_in/I_sat)), None, None

class SatAbsNL_I_approxGrad(SatAbsNL_I):
    @staticmethod
    def backward(ctx, grad_output):
        I_in, = ctx.saved_tensors
        OD, I_sat = ctx.OD, ctx.I_sat
        # Return gradients.  Note as forward takes 3 arguments, we must return 3
        # gradients, however the "gradient"'s of OD and I_sat are None.
        return grad_output * torch.exp(-OD / (1 + I_in / I_sat)), None, None

class SatAbsNL_I_zeroGrad(SatAbsNL_I):
    @staticmethod
    def backward(ctx, grad_output):
        return torch.zeros(grad_output.shape).to(grad_output.device), None, None

class SatAbsNL_I_positiveGrad(SatAbsNL_I):
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None

class SatAbsNL_I_negativeGrad(SatAbsNL_I):
    @staticmethod
    def backward(ctx, grad_output):
        return -1*grad_output, None, None

class SatAbsNL_E(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, E_in, OD, I_sat):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(E_in)
        ctx.OD = OD # Note this OD is still defined w.r.t the transmitted intensity.
        ctx.I_sat = I_sat

        return E_in * torch.exp(- (OD/2) / (1 + (E_in**2)/I_sat))

    @staticmethod
    @abstractmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        raise NotImplementedError()

class SatAbsNL_E_exactGrad(SatAbsNL_E):
    @staticmethod
    def backward(ctx, grad_output):
        E_in, = ctx.saved_tensors
        OD, I_sat = ctx.OD, ctx.I_sat

        # Return gradients.  Note as forward takes 3 arguments, we must return 3
        # gradients, however the "gradient"'s of OD and I_sat are None.
        return grad_output * (1 + ((E_in**2)/I_sat)*( OD/ (1+(E_in**2)/I_sat)**2 )) * torch.exp(-(OD/2) / (1 + (E_in**2) / I_sat)), None, None

class SatAbsNL_E_approxGrad(SatAbsNL_E):
    @staticmethod
    def backward(ctx, grad_output):
        E_in, = ctx.saved_tensors
        OD, I_sat = ctx.OD, ctx.I_sat
        # Return gradients.  Note as forward takes 3 arguments, we must return 3
        # gradients, however the "gradient"'s of OD and I_sat are None.
        return grad_output * torch.exp(-(OD/2) / (1 + (E_in**2)/I_sat)), None, None

class SatAbsNL_E_zeroGrad(SatAbsNL_E):
    @staticmethod
    def backward(ctx, grad_output):
        return torch.zeros(grad_output.shape).to(grad_output.device), None, None

class SatAbsNL_E_positiveGrad(SatAbsNL_E):
    @staticmethod
    def backward(ctx, grad_output):
        return np.exp(-5)*grad_output, None, None

class SatAbsNL_E_negativeGrad(SatAbsNL_E):
    @staticmethod
    def backward(ctx, grad_output):
        return -1*grad_output, None, None