
import os
import random
import numpy as np

from keras import backend as K
import tensorflow as tf
# sess = tf.InteractiveSession()

"""
some basics on the loss functions defitntion in keras:
https://stackoverflow.com/questions/45480820/keras-how-to-get-tensor-dimensions-inside-custom-loss

INPUTS:
y_true (batch_size, n_classes), one hot vector (of size n_class) for every data point in the batch (eg 64)
y_pred (batch_size, n_classes), model predictions (of size n_class) for every data point in the batch (eg 64)

OUTPUTS:
loss is a 1D tensor (of size n_class) and not a scalar
it contains the loss for every data point.
Therefore, in keras, the loss function DOES NOT include the averaging of the convergence
The computation of the mean loss over all the batch data points (which is 1-dimensional, just the batch dimension)
is done outside the loss function definition, in training.py)


basic operations with K backend*************************************************************************************

K.cast(y_true_v, 'float32') :  casts y_true_v to the type float32

K.greater(loss, m): Element-wise truth value of (loss > m).
Returns A bool tensor.

keras.backend.clip(x, min_value, max_value)
Element-wise value clipping.

to get a tensor shape K.int_shape:
first dimension is batch dimension: int_shape(y_true)[0] will return you a batch size.
You should use int_shape(y_true)[1] for the number of classes

K.int_shape(y_true) gives (None, None)
K.int_shape(y_pred) gives (None, 26) so it looks valid.

K.ndim Returns the number of axes in a tensor, as an integer.

keras.backend.std(x, axis=None, keepdims=False) Standard deviation of a tensor, alongside the specified axis.

k.max(x, axis=None, keepdims=False)
axis: An integer or list of integers in [-rank(x), rank(x)), the axes to find maximum values.
If None (default), finds the maximum over all dimensions.


BIB:
https://stackoverflow.com/questions/43818584/custom-loss-function-in-keras

Implementation idea:

We want our loss to depend on variables beyond y_true and y_pred, but a loss function must accept only 2 arguments:
y_true and y_pred

How to workaround this?
- Create a wrapper function aka function closure (without constraints on the number or type of arguments)
- that returns a legitimate loss function (with only the 2 typical arguments y_true and y_pred),
- such that the legitimate loss function has access to the argument(s) of its enclosing function,
- which are necessary for the more complex processing

"""


"""********************************************************************************************************************
DIY modifications of cross entropy loss:
small tweaks: MAXIMUM
"""


def crossentropy_cochlear(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    supongo que eso tiene 20 reales, uno por clase. Esto es para un single
    :return:
    """
    # Explanation for the first line of code, that determines the whole function
    # sumo los valores de las 20 proba, para cada data_point. (en teoria ha de sumar 1) y lo comparo con 1.1.
    # Cuando lo supere, saco un True
    # esto en teoria nunca deberia pasar, a no ser que se haya hecho algun apanyo para marcar las verified? YESSSSSS

    # 1st line tiene sentido si y_true viene con un flag metido. Ejemplo:
    # si es un mV, y_true es un one-hot-vector, as always
    # si es nV, le meto un offset a all the values del one-hot-vector, de 1 por ejemplo (hence sum > 1)
    # esto es una manera elegante de saber de donde viene cada data point :)

    # vip:
    # 1) determine the origin of the patch, as a boolean vector in y_true_v, con 64 booleans (True = patch from nV)
    #  y_true_v = True if the patch comes from nV
    y_true_v = K.greater(K.sum(y_true, axis=-1), 1.1)

    # 2) now that I have this info, convert y_true to original format one-hot-vector. Then it can be used as always
    # x % y (remainder of x/y); make sure it is less than 1
    y_true = y_true % 1

    # make sure the predicted values are in range [epsilon: 1]
    y_pred = K.clip(y_pred, K.epsilon(), 1)

    # compute loss for every data point, ie patch
    # es 64 (batch_size) losses
    loss = -K.sum(y_true * K.log(y_pred), axis=-1)

    # --------------------------compute threshold to discard noisy patches
    # threshold m
    m = K.max(loss) * 0.8
    # loss contiene 64 losses, una por cada elemento del batch. Pilla el maximo y calcula el threshold m

    loss = loss

    el = 1 - (K.cast(K.greater(loss, m), 'float32') * K.cast(y_true_v, 'float32'))
    # vip: las dos condiciones para descartar un data point son:
    # -loss > m (K.cast(K.greater(loss, m), 'float32'))
    # -que sea non-veri (K.cast(y_true_v, 'float32'))
    # cuando ambas son True, su producto es 1, que se resta de 1, dando 0 y ese data point se descarta

    # K.greater(loss, m) es vector boolean de 64 con True solo para las entries con loss gigante
    # eso se multiplca por y_true_v, que es un boolean vector de 64

    # el es un vector binario de 64 values. Solo tiene 0 para los patches que:
    # - metian mucha loss (mas que el threshold m).
    # - origin is nV
    # lo que enmascaras son data points, ie patches

    loss = loss * el
    # element wise multiplication, de batch_size dimensions
    # loss contiene 64 losses, pero algunos de sus entries son 0, ignorando noisy-labeled patches

    return loss


def crossentropy_diy_max(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    supongo que eso tiene 20 reales, uno por clase. Esto es para un single
    :return:
    """

    # print shapes: the params are (batch_size, n_classes)
    print(K.int_shape(y_true))
    print(K.int_shape(y_pred))

    # make sure the predicted values are in range [epsilon: 1]
    y_pred = K.clip(y_pred, K.epsilon(), 1)

    # compute loss for every data point, ie patch
    # i guess axis=-1 means sum over the 20 real values for the 20 classes, returning a loss for  a data point
    loss = -K.sum(y_true * K.log(y_pred), axis=-1)
    # length is 64 (batch_size) losses
    print(K.int_shape(loss))

    # --------------------------compute threshold to discard noisy patches
    # threshold m
    m = K.max(loss) * 0.7
    # loss contiene 64 losses, una por cada elemento del batch. Pilla el maximo y calcula el threshold m

    loss = loss

    el = 1 - (K.cast(K.greater(loss, m), 'float32'))
    # vip: la unica condicion para descartar un data point es:
    # -loss > m (K.cast(K.greater(loss, m), 'float32'))
    print(K.int_shape(el))

    # K.greater(loss, m) es vector boolean de 64 con True solo para las entries con loss gigante

    # el es un vector binario de 64 values. Solo tiene 0 para los patches que:
    # - metian mucha loss (mas que el threshold m).
    # lo que enmascaras son data points, ie patches

    loss = loss * el
    # element wise multiplication, de batch_size dimensions
    # loss contiene 64 losses, pero algunos de sus entries son 0, ignorando noisy-labeled patches
    print(K.int_shape(loss))

    return loss


def crossentropy_diy_max_wrap(_r):
    def crossentropy_diy_max_core(y_true, y_pred):
        print("\ncrossentropy_diy_max_wrap.crossentropy_diy_max_core")
        print(K.int_shape(y_true))
        print(K.int_shape(y_pred))

        # hyper param
        print('r:', _r)

        # make sure the predicted values are in range [epsilon: 1]
        y_pred = K.clip(y_pred, K.epsilon(), 1)

        # compute loss for every data point, ie patch
        # i guess axis=-1 means sum over the 20 real values for the 20 classes, returning a loss for  a data point
        _loss = -K.sum(y_true * K.log(y_pred), axis=-1)
        # length is 64 (batch_size) losses
        print(K.int_shape(_loss))

        # --------------------------compute threshold to discard noisy patches
        # threshold m
        _m = K.max(_loss) * _r
        # loss contiene 64 losses, una por cada elemento del batch. Pilla el maximo y calcula el threshold m

        _el = 1 - (K.cast(K.greater(_loss, _m), 'float32'))
        # vip: la unica condicion para descartar un data point es:
        # -loss > m (K.cast(K.greater(loss, m), 'float32'))
        print(K.int_shape(_el))

        # K.greater(loss, m) es vector boolean de 64 con True solo para las entries con loss gigante

        # el es un vector binario de 64 values. Solo tiene 0 para los patches que:
        # - metian mucha loss (mas que el threshold m).
        # lo que enmascaras son data points, ie patches

        _loss = _loss * _el
        # element wise multiplication, de batch_size dimensions
        # loss contiene 64 losses, pero algunos de sus entries son 0, ignorando noisy-labeled patches
        print(K.int_shape(_loss))

        return _loss
    return crossentropy_diy_max_core


"""********************************************************************************************************************
DIY modifications of cross entropy loss:
small tweaks: outlier-based
"""


def crossentropy_diy_outlier(y_true, y_pred):
    """
    https://en.wikipedia.org/wiki/Box_plot#/media/File:Boxplot_vs_PDF.svg

    if we are able to sort here, the rest is trivial (grab two middle elements and mean)
    to sort:
    https://stackoverflow.com/questions/48096812/how-can-i-sort-the-values-in-a-custom-keras-tensorflow-loss-function

    y_true (batch_size, n_classes), one hot vector (of size n_class) for every data point in the batch (eg 64)
    y_pred (batch_size, n_classes), model predictions (of size n_class) for every data point in the batch (eg 64)

    :param y_true:
    :param y_pred:
    supongo que eso tiene 20 reales, uno por clase. Esto es para un single
    :return:
    """

    # make sure the predicted values are in range [epsilon: 1]
    y_pred = K.clip(y_pred, K.epsilon(), 1)

    # compute loss for every data point, ie patch according to the CCE
    # axis=-1 means sum over columns, ie the 20 real values for the 20 classes, returning a loss for every data point
    loss = -K.sum(y_true * K.log(y_pred), axis=-1)
    # length of loss is (64,) (batch_size,) losses
    # print(loss.eval())

    # --------------------------compute threshold to discard noisy patches
    def get_real_median(v):
        """
        given a tensor with shape (batch_size,), compute and return the median
        consider both odd and even cases

        :param v:
        :return:
        """

        # v = tf.reshape(v, [-1])
        # n_categ = v.get_shape()[0]
        # mid = n_categ // 2 + 1
        # TODO casca aqui. offline si que esta bien, pero aqui no.
        # val = tf.nn.top_k(v, mid).values
        # sort
        # val = tf.nn.top_k(v, n_categ).values
        # 33 = 64/2 + 1 ie batch_size/2 + 1. To pick the sorted top +1 points
        val = tf.nn.top_k(v, 33).values

        # print(val.eval())
        # simple: batch_size is always even
        # if n_categ % 2 == 1:
        #     return val[-1]
        # else:
        return 0.5 * (val[-1] + val[-2])

    # print(loss.eval())

    mean_loss, var_loss = tf.nn.moments(loss, axes=[0])
    # If x is 1-D and axes = [0] this is just the mean and variance of a vector.
    # if it is 2D, and axes=[-1], compute var across columns, ie for all the n_class losses of a datapoint

    median_loss = get_real_median(loss)
    # print(median_loss.eval())
    std_loss = tf.sqrt(var_loss)
    # print(std_loss.eval())

    threshold = median_loss + 2.7 * std_loss
    # print(threshold.eval())

    # outliers would be points greater than 2.7*std away from the MEDIAN of the sample data,
    # loss contiene 64 losses, una por cada elemento del batch.

    el = 1 - (K.cast(K.greater(loss, threshold), 'float32'))
    # print(el.eval())
    # vip: la unica condicion para descartar un data point es:
    # -loss > threshold
    # K.greater(loss, m) es vector boolean de 64 con True solo para las entries con loss gigante
    # el es un vector binario de 64 values. Solo tiene 0 para los patches que:
    # - metian mucha loss (mas que el threshold).
    # lo que enmascaras son data points, ie patches
    loss = loss * el
    # element wise multiplication, de batch_size dimensions
    # loss contiene 64 losses, pero algunos de sus entries son 0, ignorando noisy-labeled patches
    # print(loss.eval())

    return loss


def crossentropy_diy_outlier_wrap(_l):
    def crossentropy_diy_outlier_core(y_true, y_pred):
        print("\ncrossentropy_diy_outlier_wrap.crossentropy_diy_outlier_core")

        # hyper param
        print('l:', _l)

        print(K.int_shape(y_true))
        print(K.int_shape(y_pred))

        # make sure the predicted values are in range [epsilon: 1]
        y_pred = K.clip(y_pred, K.epsilon(), 1)

        # compute loss for every data point, ie patch according to the CCE
        # axis=-1 means sum over columns, ie the 20 real values for the 20 classes, returning a loss for every data point
        _loss = -K.sum(y_true * K.log(y_pred), axis=-1)
        # length of loss is (64,) (batch_size,) losses
        print(K.int_shape(_loss))

        # --------------------------compute threshold to discard noisy patches
        def _get_real_median(_v):
            """
            given a tensor with shape (batch_size,), compute and return the median
            consider both odd and even cases

            :param v:
            :return:
            """
            # 33 = 64/2 + 1 ie batch_size/2 + 1. To pick the sorted top +1 points
            _val = tf.nn.top_k(_v, 33).values
            return 0.5 * (_val[-1] + _val[-2])

        _mean_loss, _var_loss = tf.nn.moments(_loss, axes=[0])
        # If x is 1-D and axes = [0] this is just the mean and variance of a vector.
        _median_loss = _get_real_median(_loss)
        _std_loss = tf.sqrt(_var_loss)

        _threshold = _median_loss + _l*_std_loss
        # outliers would be points greater than 2.7*std away from the MEDIAN of the sample data,
        # loss contiene 64 losses, una por cada elemento del batch.

        _el = 1 - (K.cast(K.greater(_loss, _threshold), 'float32'))
        # vip: la unica condicion para descartar un data point es:
        # -loss > m (K.cast(K.greater(loss, m), 'float32'))
        print(K.int_shape(_el))
        # -loss > threshold
        # K.greater(loss, _threshold) es vector boolean de 64 con True solo para las entries con loss gigante
        # el es un vector binario de 64 values. Solo tiene 0 para los patches que:
        # - metian mucha loss (mas que el threshold).
        # lo que enmascaras son data points, ie patches

        _loss = _loss * _el
        # element wise multiplication, de batch_size dimensions
        # loss contiene 64 losses, pero algunos de sus entries son 0, ignorando noisy-labeled patches
        print(K.int_shape(_loss))
        return _loss
    return crossentropy_diy_outlier_core


"""********************************************************************************************************************
reed
"""


def crossentropy_reed(y_true, y_pred):
    """
    :param y_true:
    :param y_pred:
    supongo que eso tiene 20 reales, uno por clase. Esto es para un single
    :return:
    """
    beta = 0.95

    # (1) dynamically update the targets based on the current state of the model: bootstrapped target tensor
    # vip soft: use repdicted class proba directly to generate regression targets
    y_true_update = beta*y_true + (1 - beta)*y_pred

    # vip hard: modify regression taret using the MAP estimate of of the predicted proba (a one-hot-vector of the predicted proba)
    zeta = (K.cast(K.greater_equal(y_pred, K.max(y_pred)), 'float32'))
    y_true_update = beta*y_true + (1 - beta)*zeta

    # (2) compute loss as always for every data point, ie patch
    # i guess axis=-1 means sum over the 20 real values for the 20 classes, returning a loss for a data point
    loss = -K.sum(y_true_update * K.log(y_pred), axis=-1)
    # length is 64 (batch_size) losses
    print(K.int_shape(loss))

    return loss


def crossentropy_reed_wrap(_type, _beta):
    """

    :param _beta:
    :param _type:
    :return:
    """
    def crossentropy_reed_core(y_true, y_pred):
        print("\ncrossentropy_reed_wrap.crossentropy_reed_core")

        # hyper param
        print('type:', _type)
        print('beta:', _beta)

        # make sure the predicted values are in range [epsilon: 1]
        y_pred = K.clip(y_pred, K.epsilon(), 1)

        # (1) dynamically update the targets based on the current state of the model: bootstrapped target tensor
        if _type == 'soft':
            # vip soft: use repdicted class proba directly to generate regression targets
            print(K.int_shape(y_pred))
            y_true_update = _beta * y_true + (1 - _beta) * y_pred
            print(K.int_shape(y_true_update))

        elif _type == 'hard':
            # vip hard: modify regression taret using the MAP estimate of of the predicted proba (a one-hot-vector of the predicted proba)
            zeta = (K.cast(K.greater_equal(y_pred, K.max(y_pred)), 'float32'))
            y_true_update = _beta * y_true + (1 - _beta) * zeta
            K.print_tensor(zeta, message="zeta is: ")

        # (2) compute loss as always for every data point, ie patch
        # i guess axis=-1 means sum over the 20 real values for the 20 classes, returning a loss for a data point
        _loss = -K.sum(y_true_update * K.log(y_pred), axis=-1)
        # length is 64 (batch_size) losses
        # print(K.int_shape(_loss))
        return _loss
    return crossentropy_reed_core



"""********************************************************************************************************************
generalization of cross entropy loss:
https://arxiv.org/pdf/1805.07836.pdf
pytorch code provided by Zhang
"""


# pytorch code
# def Lq_loss(outputs, labels, q, num_classes = 10):
#
#     # apply a softmax layer on the net outputs (done previously): y_pred
#     outputs = F.softmax(outputs, dim=1)
#
#     # convert your labels to one_hot encoding (done previously): y_true
#     one_hot = Variable(torch.zeros(labels.size(0), num_classes).cuda().scatter_(1, labels.long().view(-1, 1).data, 1))
#
#     # returns one on the places of elements greater than 0, element-wise. y_true
#     mask = one_hot.gt(0)
#
#     # masked_select: Returns a new 1-D tensor which indexes the input tensor according to the binary mask
#     # ie, superimpose mask over outputs and return the result in 1D tensor
#     # multi-class problem: the y_true has only one active class
#     # hence we focus on the probabilitiy for the correct class only (disregard the others).
#     # dont care how proba in the incorrect classes are spread
#     loss = torch.masked_select(outputs, mask)
#     # either loss is a real number (one patch) or 64 real numbers (entire mini batch)
#
#     # compute the Lq loss between the one-hot encoded label and your output (equation 6 of our paper).
#     loss = (1 - (loss+10**(-8))**q) / q
#
#     # compute average of divergence across the mini batch
#     # loss.shape[0] indicates that there is more than one element, hence all the patches in a batch
#     loss = loss.sum() / loss.shape[0]
#
#     return loss


def lq_loss(y_true, y_pred):

    print(K.int_shape(y_true))
    print(K.int_shape(y_pred))

    # hyper param
    q = 0.7

    # masked_select: Returns a new 1-D tensor which indexes the input tensor according to the binary mask
    # ie, superimpose mask over outputs and return the result in 1D tensor
    # multi-class problem: the y_true has only one active class
    # hence we focus on the probabilitiy for the correct class only (disregard the others).
    # dont care how proba in the incorrect classes are spread
    # loss = torch.masked_select(outputs, mask)
    # either loss is a real number (one patch) or vip 64 real numbers (entire mini batch)

    # keeping the dimensions, but the elements !=0 in tensor tmp are only those corresponding to the target classes
    tmp = y_pred * y_true
    print(K.int_shape(tmp))

    # grab maximum value (only one !=0) from the n_class values of each data instance (patch)
    loss = K.max(tmp, axis=-1)
    print(K.int_shape(loss))

    # compute the Lq loss between the one-hot encoded label and your output (equation 6 of paper).
    loss = (1 - (loss + 10**(-8))**q) / q
    print(K.int_shape(loss))

    # compute average of divergence across the mini batch
    # loss.shape[0] indicates that there is more than one element, hence all the patches in a batch
    # vip in keras this is done outside this function. we output a 1D tensor of n_class elements (and not a scalar)
    # loss = loss.sum() / loss.shape[0]

    return loss


# inspired by https://github.com/keras-team/keras/issues/2121#issuecomment-214551349
# def penalized_loss(noise):
#     def loss(y_true, y_pred):
#         return K.mean(K.square(y_pred - y_true) - K.square(y_true - noise), axis=-1)
#     return loss
# model.compile(loss=[penalized_loss(noise=output2), penalized_loss(noise=output1)], optimizer='rmsprop')


def lq_loss_wrap(_q):
    def lq_loss_core(y_true, y_pred):
        print("\nlq_loss_wrap.lq_loss_core")
        print(K.int_shape(y_true))
        print(K.int_shape(y_pred))

        # hyper param
        print('q:', _q)

        # keeping the dimensions, but the elements !=0 in tensor tmp are only those corresponding to the target classes
        _tmp = y_pred * y_true
        print(K.int_shape(_tmp))

        # grab maximum value (only one !=0) from the n_class values of each data instance (patch)
        _loss = K.max(_tmp, axis=-1)
        print(K.int_shape(_loss))

        # compute the Lq loss between the one-hot encoded label and your output (equation 6 of paper).
        _loss = (1 - (_loss + 10 ** (-8)) ** _q) / _q
        print(K.int_shape(_loss))

        # compute average of divergence across the mini batch
        # loss.shape[0] indicates that there is more than one element, hence all the patches in a batch
        # vip in keras this is done outside this function. we output a 1D tensor of n_class elements (and not a scalar)
        # loss = loss.sum() / loss.shape[0]

        return _loss
    return lq_loss_core


def lq_loss_truncated_wrap(_q, _k):
    def lq_loss_truncated_core(y_true, y_pred):
        print("\nlq_loss_truncated_wrap.lq_loss_truncated_core")
        print(K.int_shape(y_true))
        print(K.int_shape(y_pred))

        # hyper param
        print('q:', _q)
        print('k:', _k)

        # keeping the dimensions, but the elements !=0 in tensor tmp are only those corresponding to the target classes
        _tmp = y_pred * y_true
        print(K.int_shape(_tmp))

        # grab maximum value (only one !=0) from the n_class values of each data instance (patch)
        _loss = K.max(_tmp, axis=-1)
        print(K.int_shape(_loss))

        # compute the Lq loss between the one-hot encoded label and your output (equation 6 of paper).
        _loss = (1 - (_loss + 10 ** (-8)) ** _q) / _q
        print(K.int_shape(_loss))

        # compute average of divergence across the mini batch
        # loss.shape[0] indicates that there is more than one element, hence all the patches in a batch
        # watch in keras this is done outside this function. we output a 1D tensor of n_class elements (and not a scalar)
        # loss = loss.sum() / loss.shape[0]

        # vip so far we got batch_size floats (loss values) in _loss. lets do trauncated lqloss
        # calculate threshold Lqk. this is a constant
        lqk = (1 - _k ** _q) / _q
        print('lqk:', lqk)

        # create binary mask of batch_size values such that everything greater than lqk is 0 (to discard)
        _mask_discard = 1 - (K.cast(K.greater(_loss, lqk), 'float32'))
        # K.greater(loss, lqk) es vector boolean de batch_size con True para las entries con loss grande

        # la unica condicion para descartar un data point es:
        # loss > lqk (K.cast(K.greater(loss, lqk), 'float32'))

        # datapoints (patches) with large loss (more than lqk) are masked, discarded.
        # element wise multiplication, batch_size dimensions
        _loss = _loss * _mask_discard
        # some of the entries in _loss are 0 (noisy-labeled patches):
        # -do not contribute to weights update
        # -no push in the backprop

        return _loss
    return lq_loss_truncated_core



# lq loss but thresholded as I do in Lmax **************************************************************************
def lq_loss_max_wrap(_q, _r):
    def lq_loss_max_core(y_true, y_pred):
        print("\nlq_loss_max_wrap.lq_loss_max_core")
        print(K.int_shape(y_true))
        print(K.int_shape(y_pred))

        # hyper param
        print('q:', _q)
        print('r:', _r)

        # keeping the dimensions, but the elements !=0 in tensor tmp are only those corresponding to the target classes
        _tmp = y_pred * y_true
        print(K.int_shape(_tmp))

        # grab maximum value (only one !=0) from the n_class values of each data instance (patch)
        _loss = K.max(_tmp, axis=-1)
        print(K.int_shape(_loss))

        # compute the Lq loss between the one-hot encoded label and your output (equation 6 of paper).
        _loss = (1 - (_loss + 10 ** (-8)) ** _q) / _q
        print(K.int_shape(_loss))
        # length is 64 (batch_size) losses

        # --------------------------compute threshold to discard noisy patches
        # threshold m
        _m = K.max(_loss) * _r
        # loss contiene 64 losses, una por cada elemento del batch. Pilla el maximo y calcula el threshold m

        _el = 1 - (K.cast(K.greater(_loss, _m), 'float32'))
        # vip: la unica condicion para descartar un data point es:
        # -loss > m (K.cast(K.greater(loss, m), 'float32'))
        print(K.int_shape(_el))

        # K.greater(loss, m) es vector boolean de 64 con True solo para las entries con loss gigante

        # el es un vector binario de 64 values. Solo tiene 0 para los patches que:
        # - metian mucha loss (mas que el threshold m).
        # lo que enmascaras son data points, ie patches

        _loss = _loss * _el
        # element wise multiplication, de batch_size dimensions
        # loss contiene 64 losses, pero algunos de sus entries son 0, ignorando noisy-labeled patches
        print(K.int_shape(_loss))

        return _loss
    return lq_loss_max_core


# lq loss but thresholded as I do in Loutlier **************************************************************************
def lq_loss_outlier_wrap(_q, _l):
    def lq_loss_outlier_core(y_true, y_pred):
        print("\nlq_loss_outlier_wrap.lq_loss_outlier_core")
        print(K.int_shape(y_true))
        print(K.int_shape(y_pred))

        # hyper param
        print('q:', _q)
        print('l:', _l)

        # keeping the dimensions, but the elements !=0 in tensor tmp are only those corresponding to the target classes
        _tmp = y_pred * y_true
        print(K.int_shape(_tmp))

        # grab maximum value (only one !=0) from the n_class values of each data instance (patch)
        _loss = K.max(_tmp, axis=-1)
        print(K.int_shape(_loss))

        # compute the Lq loss between the one-hot encoded label and your output (equation 6 of paper).
        _loss = (1 - (_loss + 10 ** (-8)) ** _q) / _q
        print(K.int_shape(_loss))
        # length is 64 (batch_size) losses

        # --------------------------compute threshold to discard noisy patches
        def _get_real_median(_v):
            """
            given a tensor with shape (batch_size,), compute and return the median
            consider both odd and even cases

            :param v:
            :return:
            """
            # 33 = 64/2 + 1 ie batch_size/2 + 1. To pick the sorted top +1 points
            _val = tf.nn.top_k(_v, 33).values
            return 0.5 * (_val[-1] + _val[-2])

        _mean_loss, _var_loss = tf.nn.moments(_loss, axes=[0])
        # If x is 1-D and axes = [0] this is just the mean and variance of a vector.
        _median_loss = _get_real_median(_loss)
        _std_loss = tf.sqrt(_var_loss)

        _threshold = _median_loss + _l*_std_loss
        # outliers would be points greater than 2.7*std away from the MEDIAN of the sample data,
        # loss contiene 64 losses, una por cada elemento del batch.

        _el = 1 - (K.cast(K.greater(_loss, _threshold), 'float32'))
        # vip: la unica condicion para descartar un data point es:
        # -loss > m (K.cast(K.greater(loss, m), 'float32'))
        print(K.int_shape(_el))
        # -loss > threshold
        # K.greater(loss, _threshold) es vector boolean de 64 con True solo para las entries con loss gigante
        # el es un vector binario de 64 values. Solo tiene 0 para los patches que:
        # - metian mucha loss (mas que el threshold).
        # lo que enmascaras son data points, ie patches

        _loss = _loss * _el
        # element wise multiplication, de batch_size dimensions
        # loss contiene 64 losses, pero algunos de sus entries son 0, ignorando noisy-labeled patches
        print(K.int_shape(_loss))
        return _loss
    return lq_loss_outlier_core




# check this for tf implementation of reed
# https://github.com/tensorflow/models/blob/master/research/object_detection/core/losses.py
        # def _compute_loss(self, prediction_tensor, target_tensor, weights):
  #   """Compute loss function.
  #   Args:
  #     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
  #       num_classes] representing the predicted logits for each class
  #     target_tensor: A float tensor of shape [batch_size, num_anchors,
  #       num_classes] representing one-hot encoded classification targets
  #     weights: a float tensor of shape [batch_size, num_anchors]
  #   Returns:
  #     loss: a float tensor of shape [batch_size, num_anchors, num_classes]
  #       representing the value of the loss function.
  #   """
    # if self._bootstrap_type == 'soft':
      # bootstrap_target_tensor = self._alpha * target_tensor + (
      #     1.0 - self._alpha) * tf.sigmoid(prediction_tensor)
    # else:
      # bootstrap_target_tensor = self._alpha * target_tensor + (
      #     1.0 - self._alpha) * tf.cast(
      #         tf.sigmoid(prediction_tensor) > 0.5, tf.float32)


    # compute whatever loss function, based on the bootstrapped target tensor
    # per_entry_cross_ent = (tf.nn.sigmoid_cross_entropy_with_logits(
    #     labels=bootstrap_target_tensor, logits=prediction_tensor))

    # return per_entry_cross_ent * tf.expand_dims(weights, 2)



"""
from here on we distinguish data points in the batch, based on its origin
we only apply robustness measures if they belong from the noisy subset
we are being selective here.
hoping this makes the most out of both subsets
-discard the losses of some data points (max n outlier)
-applying beta =1 in reed (hence no convex combination of targets)
-applying CCE in Zhang (instead of the lq_loss

using masks it is easy
"""


"""
DIY modifications of cross entropy loss:
small tweaks: max-based considering clip origin (this makes sense when clean set and noisy set are considered
"""


def crossentropy_diy_max_origin(y_true, y_pred):
    """
    version for simulations and testing
    :param y_true:
    :param y_pred:
    supongo que eso tiene 20 reales, uno por clase. Esto es para un single
    :return:
    """

    print(y_true)
    print(K.int_shape(y_true))
    print(y_pred)
    print(K.int_shape(y_pred))

    # 1st line tiene sentido si y_true viene con un flag metido. Ejemplo:
    # si es un mV, y_true es un one-hot-vector, as always
    # si es nV, le meto un offset a all the values del one-hot-vector, (hence sum > 1)
    # esto es una manera elegante de saber de donde viene cada data point :)

    # 1) determine the origin of the patch, as a boolean vector in y_true_flag, con batch_size booleans
    # (True = patch from noisy subset)
    # y_true_flag will be used later as an additional condition to discard samples
    y_true_flag = K.greater(K.sum(y_true, axis=-1), 90)
    print(y_true_flag)
    print(K.int_shape(y_true_flag))

    print('\nmasks:')
    # 2) now that I have this info, convert convert the input y_true (with flags inside) into
    # a valid y_true one-hot-vector format. Then it can be used as always

    # attenuating factor for data points that need it (those that came with a one-hot of 100)
    mask_reduce = K.cast(y_true_flag, 'float32') * 0.01
    print(mask_reduce)
    print(K.int_shape(mask_reduce))

    # identity factor for standard one-hot vectors
    mask_keep = K.cast(K.equal(y_true_flag, False), 'float32')
    print(mask_keep)
    print(K.int_shape(mask_keep))

    # combine 2 masks
    mask = mask_reduce + mask_keep
    print(mask)
    print(K.int_shape(mask))

    # current shape is (None,) need shape of y_true or else Incompatible shapes: [64,20] vs. [64]
    y_true_shape = K.shape(y_true)
    # fix rows with batch_size, given by y_true_shape[0]
    mask = K.reshape(mask, (y_true_shape[0], 1))
    print(mask)
    print(K.int_shape(mask))

    # todo fix columns with n_classes, hardcoded, yielding final mask
    # mask_total = K.repeat_elements(mask, 20, axis=1)
    # print(mask_total)
    # print(K.int_shape(mask_total))

    print('\napplying mask:') # to have a valid y_true TODO total or mask?
    # y_true = y_true * mask_total
    y_true = y_true * mask
    print(y_true)
    print(K.int_shape(y_true))

    # make sure the true values are in range [epsilon: 1] after the hassle
    y_true = K.clip(y_true, K.epsilon(), 1)
    # make sure the predicted values are in range [epsilon: 1]
    y_pred = K.clip(y_pred, K.epsilon(), 1)


    # compute loss for every data point, ie patch
    # es 64 (batch_size) losses
    loss = -K.sum(y_true * K.log(y_pred), axis=-1)

    # --------------------------compute threshold to discard noisy patches
    # threshold m
    m = K.max(loss) * 0.8
    # loss contiene 64 losses, una por cada elemento del batch. Pilla el maximo y calcula el threshold m


    el = 1 - (K.cast(K.greater(loss, m), 'float32') * K.cast(y_true_flag, 'float32'))
    # vip: las dos condiciones para descartar un data point son:
    # -loss > m (K.cast(K.greater(loss, m), 'float32'))
    # -que sea non-veri (K.cast(y_true_v, 'float32'))
    # cuando ambas son True, su producto es 1, que se resta de 1, dando 0 y ese data point se descarta

    # K.greater(loss, m) es vector boolean de 64 con True solo para las entries con loss gigante
    # eso se multiplca por y_true_v, que es un boolean vector de 64

    # el es un vector binario de 64 values. Solo tiene 0 para los patches que:
    # - metian mucha loss (mas que el threshold m).
    # - origin is nV
    # lo que enmascaras son data points, ie patches

    loss = loss * el
    # element wise multiplication, de batch_size dimensions
    # loss contiene 64 losses, pero algunos de sus entries son 0, ignorando noisy-labeled patches

    return loss


def crossentropy_diy_max_origin_wrap(_r):
    """
    version for experiments
    :param _r:
    :return:
    """
    def crossentropy_diy_max_origin_core(y_true, y_pred):

        # hyper param
        print(_r)

        print(y_true)
        print(K.int_shape(y_true))
        print(y_pred)
        print(K.int_shape(y_pred))

        # 1st line tiene sentido si y_true viene con un flag metido. Ejemplo:
        # si es un mV, y_true es un one-hot-vector, as always
        # si es nV, le meto un offset a all the values del one-hot-vector, (hence sum > 1)
        # esto es una manera elegante de saber de donde viene cada data point :)

        # 1) determine the origin of the patch, as a boolean vector in y_true_flag, con batch_size booleans
        # (True = patch from noisy subset)
        # y_true_flag will be used later as an additional condition to discard samples
        _y_true_flag = K.greater(K.sum(y_true, axis=-1), 90)

        # 2) now that I have this info, convert convert the input y_true (with flags inside) into
        # a valid y_true one-hot-vector format. Then it can be used as always

        # attenuating factor for data points that need it (those that came with a one-hot of 100)
        _mask_reduce = K.cast(_y_true_flag, 'float32') * 0.01

        # identity factor for standard one-hot vectors
        _mask_keep = K.cast(K.equal(_y_true_flag, False), 'float32')

        # combine 2 masks
        _mask = _mask_reduce + _mask_keep

        # current shape is (None,) need shape of y_true or else Incompatible shapes: [64,20] vs. [64]
        _y_true_shape = K.shape(y_true)
        # fix rows with batch_size, given by y_true_shape[0]
        _mask = K.reshape(_mask, (_y_true_shape[0], 1))

        # todo fix columns with n_classes, hardcoded, yielding final mask
        # mask_total = K.repeat_elements(mask, 20, axis=1)
        # print(mask_total)
        # print(K.int_shape(mask_total))

        # vip applying mask to have a valid y_true that we can use as always TODO total or mask?
        # y_true = y_true * mask_total
        y_true = y_true * _mask

        # make sure the true values are in range [epsilon: 1] after the hassle
        y_true = K.clip(y_true, K.epsilon(), 1)
        # make sure the predicted values are in range [epsilon: 1]
        y_pred = K.clip(y_pred, K.epsilon(), 1)

        # compute loss for every data point, ie patch
        # es 64 (batch_size) losses
        _loss = -K.sum(y_true * K.log(y_pred), axis=-1)

        # --------------------------compute threshold to discard noisy patches
        # threshold m
        _m = K.max(_loss) * _r
        # loss contiene 64 losses, una por cada elemento del batch. Pilla el maximo y calcula el threshold m

        _el = 1 - (K.cast(K.greater(_loss, _m), 'float32') * K.cast(_y_true_flag, 'float32'))
        # vip: las dos condiciones para descartar un data point son:
        # -loss > m (K.cast(K.greater(loss, m), 'float32'))
        # -que sea non-veri (K.cast(y_true_v, 'float32'))
        # cuando ambas son True, su producto es 1, que se resta de 1, dando 0 y ese data point se descarta

        # K.greater(loss, m) es vector boolean de 64 con True solo para las entries con loss gigante
        # eso se multiplca por y_true_v, que es un boolean vector de 64

        # el es un vector binario de 64 values. Solo tiene 0 para los patches que:
        # - metian mucha loss (mas que el threshold m).
        # - origin is nV
        # lo que enmascaras son data points, ie patches

        _loss = _loss * _el
        # element wise multiplication, de batch_size dimensions
        # loss contiene 64 losses, pero algunos de sus entries son 0, ignorando noisy-labeled patches

        return _loss
    return crossentropy_diy_max_origin_core


"""
DIY modifications of cross entropy loss:
small tweaks: outlier-based considering clip origin (this makes sense when clean set and noisy set are considered
"""


def crossentropy_diy_outlier_origin_wrap(_l):
    def crossentropy_diy_outlier_origin_core(y_true, y_pred):
        print(K.int_shape(y_true))
        print(K.int_shape(y_pred))

        # hyper param
        print(_l)

        print(y_true)
        print(K.int_shape(y_true))
        print(y_pred)
        print(K.int_shape(y_pred))

        # 1st line tiene sentido si y_true viene con un flag metido. Ejemplo:
        # si es un mV, y_true es un one-hot-vector, as always
        # si es nV, le meto un offset a all the values del one-hot-vector, (hence sum > 1)
        # esto es una manera elegante de saber de donde viene cada data point :)

        # 1) determine the origin of the patch, as a boolean vector in y_true_flag, con batch_size booleans
        # (True = patch from noisy subset)
        # y_true_flag will be used later as an additional condition to discard samples
        _y_true_flag = K.greater(K.sum(y_true, axis=-1), 90)

        # 2) now that I have this info, convert convert the input y_true (with flags inside) into
        # a valid y_true one-hot-vector format. Then it can be used as always

        # attenuating factor for data points that need it (those that came with a one-hot of 100)
        _mask_reduce = K.cast(_y_true_flag, 'float32') * 0.01

        # identity factor for standard one-hot vectors
        _mask_keep = K.cast(K.equal(_y_true_flag, False), 'float32')

        # combine 2 masks
        _mask = _mask_reduce + _mask_keep

        # current shape is (None,) need shape of y_true or else Incompatible shapes: [64,20] vs. [64]
        _y_true_shape = K.shape(y_true)
        # fix rows with batch_size, given by y_true_shape[0]
        _mask = K.reshape(_mask, (_y_true_shape[0], 1))

        # todo fix columns with n_classes, hardcoded, yielding final mask
        # mask_total = K.repeat_elements(mask, 20, axis=1)
        # print(mask_total)
        # print(K.int_shape(mask_total))

        # vip applying mask to have a valid y_true that we can use as always TODO total or mask?
        # y_true = y_true * mask_total
        y_true = y_true * _mask

        # make sure the true values are in range [epsilon: 1] after the hassle
        y_true = K.clip(y_true, K.epsilon(), 1)
        # make sure the predicted values are in range [epsilon: 1]
        y_pred = K.clip(y_pred, K.epsilon(), 1)

        # compute loss for every data point, ie patch according to the CCE
        # axis=-1 means sum over columns, ie the 20 real values for the 20 classes, returning a loss for every data point
        _loss = -K.sum(y_true * K.log(y_pred), axis=-1)

        # --------------------------compute threshold to discard noisy patches
        def _get_real_median(_v):
            """
            given a tensor with shape (batch_size,), compute and return the median
            consider both odd and even cases

            :param v:
            :return:
            """
            # 33 = 64/2 + 1 ie batch_size/2 + 1. To pick the sorted top +1 points
            _val = tf.nn.top_k(_v, 33).values
            return 0.5 * (_val[-1] + _val[-2])

        _mean_loss, _var_loss = tf.nn.moments(_loss, axes=[0])
        # If x is 1-D and axes = [0] this is just the mean and variance of a vector.
        _median_loss = _get_real_median(_loss)
        _std_loss = tf.sqrt(_var_loss)

        _threshold = _median_loss + _l*_std_loss
        # outliers would be points greater than 2.7*std away from the MEDIAN of the sample data,
        # loss contiene 64 losses, una por cada elemento del batch.

        _el = 1 - (K.cast(K.greater(_loss, _threshold), 'float32') * K.cast(_y_true_flag, 'float32'))
        # vip: la unica condicion para descartar un data point es:
        # -loss > m (K.cast(K.greater(loss, m), 'float32'))
        # print(K.int_shape(_el))
        # -loss > threshold
        # K.greater(loss, _threshold) es vector boolean de 64 con True solo para las entries con loss gigante
        # el es un vector binario de 64 values. Solo tiene 0 para los patches que:
        # - metian mucha loss (mas que el threshold).
        # lo que enmascaras son data points, ie patches

        _loss = _loss * _el
        # element wise multiplication, de batch_size dimensions
        # loss contiene 64 losses, pero algunos de sus entries son 0, ignorando noisy-labeled patches
        # print(K.int_shape(_loss))
        return _loss
    return crossentropy_diy_outlier_origin_core


"""
=====================================================================================================================
generalization of cross entropy loss:
https://arxiv.org/pdf/1805.07836.pdf
pytorch code provided by Zhang
"""


def lq_loss_origin_wrap(_q):
    def lq_loss_origin_core(y_true, y_pred):
        print(K.int_shape(y_true))
        print(K.int_shape(y_pred))

        # hyper param
        print(_q)

        # 1st line tiene sentido si y_true viene con un flag metido. Ejemplo:
        # si es un mV, y_true es un one-hot-vector, as always
        # si es nV, le meto un offset a all the values del one-hot-vector, (hence sum > 1)
        # esto es una manera elegante de saber de donde viene cada data point :)

        # 1) determine the origin of the patch, as a boolean vector in y_true_flag, con batch_size booleans
        # (True = patch from noisy subset)
        # y_true_flag will be used later as an additional condition to discard samples
        _y_true_flag = K.greater(K.sum(y_true, axis=-1), 90)

        # 2) now that I have this info, convert convert the input y_true (with flags inside) into
        # a valid y_true one-hot-vector format. Then it can be used as always

        # attenuating factor for data points that need it (those that came with a one-hot of 100)
        _mask_reduce = K.cast(_y_true_flag, 'float32') * 0.01

        # identity factor for standard one-hot vectors
        _mask_keep = K.cast(K.equal(_y_true_flag, False), 'float32')

        # combine 2 masks
        _mask = _mask_reduce + _mask_keep

        # current shape is (None,) need shape of y_true or else Incompatible shapes: [64,20] vs. [64]
        _y_true_shape = K.shape(y_true)
        # fix rows with batch_size, given by y_true_shape[0]
        _mask = K.reshape(_mask, (_y_true_shape[0], 1))

        # todo fix columns with n_classes, hardcoded, yielding final mask
        # mask_total = K.repeat_elements(mask, 20, axis=1)
        # print(mask_total)
        # print(K.int_shape(mask_total))

        # vip applying mask to have a valid y_true that we can use as always TODO total or mask?
        # y_true = y_true * mask_total
        y_true = y_true * _mask

        # make sure the true values are in range [epsilon: 1] after the hassle
        y_true = K.clip(y_true, K.epsilon(), 1)
        # make sure the predicted values are in range [epsilon: 1]
        y_pred = K.clip(y_pred, K.epsilon(), 1)

        # vip until here the preparation of y_true for the rest
        # vip: now compute two types of losses, for all the data points

        # (1) compute CCE loss for every data point, ie patch
        # 64 (batch_size) losses
        _loss_CCE = -K.sum(y_true * K.log(y_pred), axis=-1)
        print(K.int_shape(_loss_CCE))

        # (2) compute lq_loss for every data point, ie patch
        # keeping the dimensions, but the elements !=0 in tensor tmp are only those corresponding to the target classes
        _tmp = y_pred * y_true
        print(K.int_shape(_tmp))

        # grab maximum value (only one !=0) from the n_class values of each data instance (patch)
        _loss_tmp = K.max(_tmp, axis=-1)
        print(K.int_shape(_loss_tmp))

        # compute the Lq loss between the one-hot encoded label and your output (equation 6 of paper).
        _loss_q = (1 - (_loss_tmp + 10 ** (-8)) ** _q) / _q
        print(K.int_shape(_loss_q))
        # 64 (batch_size) losses

        # vip: now decide which loss to take for each datapoint
        _mask_noisy = K.cast(_y_true_flag, 'float32')                   # only allows patches from noisy set
        _mask_clean = K.cast(K.equal(_y_true_flag, False), 'float32')   # only allows patches from clean set

        # points coming from clean set contribute with CCE loss
        # points coming from noisy set contribute with lq_loss
        _loss_final = _loss_CCE * _mask_clean + _loss_q * _mask_noisy

        # compute average of divergence across the mini batch
        # loss.shape[0] indicates that there is more than one element, hence all the patches in a batch
        # vip in keras this is done outside this function. we output a 1D tensor of n_class elements (and not a scalar)
        # loss = loss.sum() / loss.shape[0]

        return _loss_final
    return lq_loss_origin_core



def crossentropy_reed_origin_wrap(_type, _beta):
    """

    :param _beta:
    :param _type:
    :return:
    """
    def crossentropy_reed_origin_core(y_true, y_pred):
        # hyper param
        print(_type)
        print(_beta)

        # 1st line tiene sentido si y_true viene con un flag metido. Ejemplo:
        # si es un mV, y_true es un one-hot-vector, as always
        # si es nV, le meto un offset a all the values del one-hot-vector, (hence sum > 1)
        # esto es una manera elegante de saber de donde viene cada data point :)

        # 1) determine the origin of the patch, as a boolean vector in y_true_flag, con batch_size booleans
        # (True = patch from noisy subset)
        # y_true_flag will be used later as an additional condition to discard samples
        _y_true_flag = K.greater(K.sum(y_true, axis=-1), 90)

        # 2) now that I have this info, convert convert the input y_true (with flags inside) into
        # a valid y_true one-hot-vector format. Then it can be used as always

        # attenuating factor for data points that need it (those that came with a one-hot of 100)
        _mask_reduce = K.cast(_y_true_flag, 'float32') * 0.01

        # identity factor for standard one-hot vectors
        _mask_keep = K.cast(K.equal(_y_true_flag, False), 'float32')

        # combine 2 masks
        _mask = _mask_reduce + _mask_keep

        # current shape is (None,) need shape of y_true or else Incompatible shapes: [64,20] vs. [64]
        _y_true_shape = K.shape(y_true)
        # fix rows with batch_size, given by y_true_shape[0]. Now the shape is (None,1)
        _mask = K.reshape(_mask, (_y_true_shape[0], 1))

        # todo fix columns with n_classes, hardcoded, yielding final mask
        # mask_total = K.repeat_elements(mask, 20, axis=1)
        # print(mask_total)
        # print(K.int_shape(mask_total))

        # vip applying mask to have a valid y_true that we can use as always TODO total or mask?
        # y_true = y_true * mask_total
        y_true = y_true * _mask

        # make sure the true values are in range [epsilon: 1] after the hassle
        y_true = K.clip(y_true, K.epsilon(), 1)
        # make sure the predicted values are in range [epsilon: 1]
        y_pred = K.clip(y_pred, K.epsilon(), 1)
        # vip until here the preparation of y_true for the rest

        # (1) dynamically update the targets based on the current state of the model: bootstrapped target tensor
        if _type == 'soft':
            # vip soft: use repdicted class proba directly to generate regression targets
            print(K.int_shape(y_pred))
            y_true_bootstrapped = _beta * y_true + (1 - _beta) * y_pred
            print(K.int_shape(y_true_bootstrapped))

        elif _type == 'hard':
            # vip hard: modify regression taret using the MAP estimate of of the predicted proba (a one-hot-vector of the predicted proba)
            zeta = (K.cast(K.greater_equal(y_pred, K.max(y_pred)), 'float32'))
            y_true_bootstrapped = _beta * y_true + (1 - _beta) * zeta
            K.print_tensor(zeta, message="zeta is: ")

        # at this point we have 2 versions of y_true, both have dim (batch_size, 20)
        # - the original y_true
        # - y_true_update, with the Reed version of the target label
        # vip: now decide which target label to use for each datapoint
        _mask_noisy = K.cast(_y_true_flag, 'float32')                   # only allows patches from noisy set
        _mask_clean = K.cast(K.equal(_y_true_flag, False), 'float32')   # only allows patches from clean set

        # points coming from clean set carry the standard true one-hot vector. dim is (batch_size, 1)
        # points coming from noisy set carry the Reed bootstrapped target tensor
        # todo: trusting that the entire row is multiplied by the mask value
        # vip: there is a problem with dimensions, lets try to give the column dim to the masks
        _mask_noisy = K.reshape(_mask_noisy, (_y_true_shape[0], 1))
        _mask_clean = K.reshape(_mask_clean, (_y_true_shape[0], 1))

        y_true_final = y_true * _mask_clean + y_true_bootstrapped * _mask_noisy

        # (2) compute loss as always for every data point, ie patch
        # i guess axis=-1 means sum over the 20 real values for the 20 classes, returning a loss for a data point
        _loss = -K.sum(y_true_final * K.log(y_pred), axis=-1)
        # length is 64 (batch_size) losses
        # print(K.int_shape(_loss))
        return _loss
    return crossentropy_reed_origin_core

