import tensorflow as tf


def to_tanh_space(x, box_min=0.0, box_max=1.0):
    mul = (box_max - box_min) / 2.0
    plus = (box_max + box_min) / 2.0
    return tf.atanh((x - plus) / mul)


def from_tanh_space(x, box_min=0.0, box_max=1.0):
    mul = (box_max - box_min) / 2.0
    plus = (box_max + box_min) / 2.0
    return tf.tanh(x) * mul + plus


def set_with_mask(x, x_other, mask):
    """ Returns a tensor similar to x 
        with all the values replaced by x_other 
        where the mask evaluates to true.
    """
    mask = tf.cast(mask, dtype=x.dtype) # true/fales -> 1/0
    ones = tf.ones_like(mask, dtype=x.dtype)
    return x * (ones - mask) + x_other * mask
