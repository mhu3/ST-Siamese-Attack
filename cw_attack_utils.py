import tensorflow as tf


def to_tanh_space(x, box_min=0.0, box_max=1.0):
    """ Map x from real value to tanh space. """
    mul = (box_max - box_min) / 2.0
    plus = (box_max + box_min) / 2.0
    return tf.atanh((x - plus) / mul)


def from_tanh_space(x, box_min=0.0, box_max=1.0):
    """ Map x from tanh space to real value. """
    mul = (box_max - box_min) / 2.0
    plus = (box_max + box_min) / 2.0
    return tf.tanh(x) * mul + plus


def set_with_mask(x, x_other, mask, assign=False):
    """ Returns a tensor similar to x 
        with all the values replaced by x_other 
        where the mask evaluates to true.
    """
    # true/fales -> 1/0
    mask = tf.cast(mask, dtype=x.dtype) 
    ones = tf.ones_like(mask, dtype=x.dtype)
    # Replace values
    if assign:
        x.assign(x * (ones - mask) + x_other * mask)
    else:
        return x * (ones - mask) + x_other * mask
