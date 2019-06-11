# Copyright 2019 RnD at Spoon Radio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""SpecAugment Implementation for Tensorflow.
Related paper : https://arxiv.org/pdf/1904.08779.pdf
In this paper, show summarized parameters by each open datasets in Tabel 1.
-----------------------------------------
Policy | W  | F  | m_F |  T  |  p  | m_T
-----------------------------------------
None   |  0 |  0 |  -  |  0  |  -  |  -
-----------------------------------------
LB     | 80 | 27 |  1  | 100 | 1.0 | 1
-----------------------------------------
LD     | 80 | 27 |  2  | 100 | 1.0 | 2
-----------------------------------------
SM     | 40 | 15 |  2  |  70 | 0.2 | 2
-----------------------------------------
SS     | 40 | 27 |  2  |  70 | 0.2 | 2
-----------------------------------------
LB : LibriSpeech basic
LD : LibriSpeech double
SM : Switchboard mild
SS : Switchboard strong
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import librosa
import librosa.display
import tensorflow as tf
# from tensorflow.contrib.image import sparse_image_warp
import numpy as np
import random
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# /Users/edufonseca/miniconda3/envs/kaggle18/lib/python3.6/site-packages/tensorflow/contrib/image/python/ops/interpolate_spline.py

# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Polyharmonic spline interpolation."""


import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops

EPSILON = 0.0000000001


def _cross_squared_distance_matrix(x, y):
  """Pairwise squared distance between two (batch) matrices' rows (2nd dim).

  Computes the pairwise distances between rows of x and rows of y
  Args:
    x: [batch_size, n, d] float `Tensor`
    y: [batch_size, m, d] float `Tensor`

  Returns:
    squared_dists: [batch_size, n, m] float `Tensor`, where
    squared_dists[b,i,j] = ||x[b,i,:] - y[b,j,:]||^2
  """
  x_norm_squared = math_ops.reduce_sum(math_ops.square(x), 2)
  y_norm_squared = math_ops.reduce_sum(math_ops.square(y), 2)

  # Expand so that we can broadcast.
  x_norm_squared_tile = array_ops.expand_dims(x_norm_squared, 2)
  y_norm_squared_tile = array_ops.expand_dims(y_norm_squared, 1)

  x_y_transpose = math_ops.matmul(x, y, adjoint_b=True)

  # squared_dists[b,i,j] = ||x_bi - y_bj||^2 = x_bi'x_bi- 2x_bi'x_bj + x_bj'x_bj
  squared_dists = x_norm_squared_tile - 2 * x_y_transpose + y_norm_squared_tile

  return squared_dists


def _pairwise_squared_distance_matrix(x):
  """Pairwise squared distance among a (batch) matrix's rows (2nd dim).

  This saves a bit of computation vs. using _cross_squared_distance_matrix(x,x)

  Args:
    x: `[batch_size, n, d]` float `Tensor`

  Returns:
    squared_dists: `[batch_size, n, n]` float `Tensor`, where
    squared_dists[b,i,j] = ||x[b,i,:] - x[b,j,:]||^2
  """

  x_x_transpose = math_ops.matmul(x, x, adjoint_b=True)
  x_norm_squared = array_ops.matrix_diag_part(x_x_transpose)
  x_norm_squared_tile = array_ops.expand_dims(x_norm_squared, 2)

  # squared_dists[b,i,j] = ||x_bi - x_bj||^2 = x_bi'x_bi- 2x_bi'x_bj + x_bj'x_bj
  squared_dists = x_norm_squared_tile - 2 * x_x_transpose + array_ops.transpose(
      x_norm_squared_tile, [0, 2, 1])

  return squared_dists


def _solve_interpolation(train_points, train_values, order,
                         regularization_weight):
  """Solve for interpolation coefficients.

  Computes the coefficients of the polyharmonic interpolant for the 'training'
  data defined by (train_points, train_values) using the kernel phi.

  Args:
    train_points: `[b, n, d]` interpolation centers
    train_values: `[b, n, k]` function values
    order: order of the interpolation
    regularization_weight: weight to place on smoothness regularization term

  Returns:
    w: `[b, n, k]` weights on each interpolation center
    v: `[b, d, k]` weights on each input dimension
  """

  b, n, d = train_points.get_shape().as_list()
  _, _, k = train_values.get_shape().as_list()

  # First, rename variables so that the notation (c, f, w, v, A, B, etc.)
  # follows https://en.wikipedia.org/wiki/Polyharmonic_spline.
  # To account for python style guidelines we use
  # matrix_a for A and matrix_b for B.

  c = train_points
  f = train_values

  # Next, construct the linear system.
  with ops.name_scope('construct_linear_system'):

    matrix_a = _phi(_pairwise_squared_distance_matrix(c), order)  # [b, n, n]
    if regularization_weight > 0:
      batch_identity_matrix = np.expand_dims(np.eye(n), 0)
      batch_identity_matrix = constant_op.constant(
          batch_identity_matrix, dtype=train_points.dtype)

      matrix_a += regularization_weight * batch_identity_matrix

    # Append ones to the feature values for the bias term in the linear model.
    ones = array_ops.ones([b, n, 1], train_points.dtype)
    matrix_b = array_ops.concat([c, ones], 2)  # [b, n, d + 1]

    # [b, n + d + 1, n]
    left_block = array_ops.concat(
        [matrix_a, array_ops.transpose(matrix_b, [0, 2, 1])], 1)

    num_b_cols = matrix_b.get_shape()[2]  # d + 1
    lhs_zeros = array_ops.zeros([b, num_b_cols, num_b_cols], train_points.dtype)
    right_block = array_ops.concat([matrix_b, lhs_zeros],
                                   1)  # [b, n + d + 1, d + 1]
    lhs = array_ops.concat([left_block, right_block],
                           2)  # [b, n + d + 1, n + d + 1]

    rhs_zeros = array_ops.zeros([b, d + 1, k], train_points.dtype)
    rhs = array_ops.concat([f, rhs_zeros], 1)  # [b, n + d + 1, k]

  # Then, solve the linear system and unpack the results.
  with ops.name_scope('solve_linear_system'):
    w_v = linalg_ops.matrix_solve(lhs, rhs)
    w = w_v[:, :n, :]
    v = w_v[:, n:, :]

  return w, v


def _apply_interpolation(query_points, train_points, w, v, order):
  """Apply polyharmonic interpolation model to data.

  Given coefficients w and v for the interpolation model, we evaluate
  interpolated function values at query_points.

  Args:
    query_points: `[b, m, d]` x values to evaluate the interpolation at
    train_points: `[b, n, d]` x values that act as the interpolation centers
                    ( the c variables in the wikipedia article)
    w: `[b, n, k]` weights on each interpolation center
    v: `[b, d, k]` weights on each input dimension
    order: order of the interpolation

  Returns:
    Polyharmonic interpolation evaluated at points defined in query_points.
  """

  batch_size = train_points.get_shape()[0].value
  num_query_points = query_points.get_shape()[1].value

  # First, compute the contribution from the rbf term.
  pairwise_dists = _cross_squared_distance_matrix(query_points, train_points)
  phi_pairwise_dists = _phi(pairwise_dists, order)

  rbf_term = math_ops.matmul(phi_pairwise_dists, w)

  # Then, compute the contribution from the linear term.
  # Pad query_points with ones, for the bias term in the linear model.
  query_points_pad = array_ops.concat([
      query_points,
      array_ops.ones([batch_size, num_query_points, 1], train_points.dtype)
  ], 2)
  linear_term = math_ops.matmul(query_points_pad, v)

  return rbf_term + linear_term


def _phi(r, order):
  """Coordinate-wise nonlinearity used to define the order of the interpolation.

  See https://en.wikipedia.org/wiki/Polyharmonic_spline for the definition.

  Args:
    r: input op
    order: interpolation order

  Returns:
    phi_k evaluated coordinate-wise on r, for k = r
  """

  # using EPSILON prevents log(0), sqrt0), etc.
  # sqrt(0) is well-defined, but its gradient is not
  with ops.name_scope('phi'):
    if order == 1:
      r = math_ops.maximum(r, EPSILON)
      r = math_ops.sqrt(r)
      return r
    elif order == 2:
      return 0.5 * r * math_ops.log(math_ops.maximum(r, EPSILON))
    elif order == 4:
      return 0.5 * math_ops.square(r) * math_ops.log(
          math_ops.maximum(r, EPSILON))
    elif order % 2 == 0:
      r = math_ops.maximum(r, EPSILON)
      return 0.5 * math_ops.pow(r, 0.5 * order) * math_ops.log(r)
    else:
      r = math_ops.maximum(r, EPSILON)
      return math_ops.pow(r, 0.5 * order)


def interpolate_spline(train_points,
                       train_values,
                       query_points,
                       order,
                       regularization_weight=0.0,
                       name='interpolate_spline'):
  r"""Interpolate signal using polyharmonic interpolation.

  The interpolant has the form
  $$f(x) = \sum_{i = 1}^n w_i \phi(||x - c_i||) + v^T x + b.$$

  This is a sum of two terms: (1) a weighted sum of radial basis function (RBF)
  terms, with the centers \\(c_1, ... c_n\\), and (2) a linear term with a bias.
  The \\(c_i\\) vectors are 'training' points. In the code, b is absorbed into v
  by appending 1 as a final dimension to x. The coefficients w and v are
  estimated such that the interpolant exactly fits the value of the function at
  the \\(c_i\\) points, the vector w is orthogonal to each \\(c_i\\), and the
  vector w sums to 0. With these constraints, the coefficients can be obtained
  by solving a linear system.

  \\(\phi\\) is an RBF, parametrized by an interpolation
  order. Using order=2 produces the well-known thin-plate spline.

  We also provide the option to perform regularized interpolation. Here, the
  interpolant is selected to trade off between the squared loss on the training
  data and a certain measure of its curvature
  ([details](https://en.wikipedia.org/wiki/Polyharmonic_spline)).
  Using a regularization weight greater than zero has the effect that the
  interpolant will no longer exactly fit the training data. However, it may be
  less vulnerable to overfitting, particularly for high-order interpolation.

  Note the interpolation procedure is differentiable with respect to all inputs
  besides the order parameter.

  Args:
    train_points: `[batch_size, n, d]` float `Tensor` of n d-dimensional
      locations. These do not need to be regularly-spaced.
    train_values: `[batch_size, n, k]` float `Tensor` of n c-dimensional values
      evaluated at train_points.
    query_points: `[batch_size, m, d]` `Tensor` of m d-dimensional locations
      where we will output the interpolant's values.
    order: order of the interpolation. Common values are 1 for
      \\(\phi(r) = r\\), 2 for \\(\phi(r) = r^2 * log(r)\\) (thin-plate spline),
       or 3 for \\(\phi(r) = r^3\\).
    regularization_weight: weight placed on the regularization term.
      This will depend substantially on the problem, and it should always be
      tuned. For many problems, it is reasonable to use no regularization.
      If using a non-zero value, we recommend a small value like 0.001.
    name: name prefix for ops created by this function

  Returns:
    `[b, m, k]` float `Tensor` of query values. We use train_points and
    train_values to perform polyharmonic interpolation. The query values are
    the values of the interpolant evaluated at the locations specified in
    query_points.
  """
  with ops.name_scope(name):
    train_points = ops.convert_to_tensor(train_points)
    train_values = ops.convert_to_tensor(train_values)
    query_points = ops.convert_to_tensor(query_points)

    # First, fit the spline to the observed data.
    with ops.name_scope('solve'):
      w, v = _solve_interpolation(train_points, train_values, order,
                                  regularization_weight)

    # Then, evaluate the spline at the query locations.
    with ops.name_scope('predict'):
      query_values = _apply_interpolation(query_points, train_points, w, v,
                                          order)

  return query_values




# /Users/edufonseca/miniconda3/envs/kaggle18/lib/python3.6/site-packages/tensorflow/contrib/image/python/ops/dense_image_warp.py
# /Users/edufonseca/miniconda3/envs/kaggle18/lib/python3.6/site-packages/tensorflow/contrib/image/python/ops/dense_image_warp.py
# /Users/edufonseca/miniconda3/envs/kaggle18/lib/python3.6/site-packages/tensorflow/contrib/image/python/ops/dense_image_warp.py
# /Users/edufonseca/miniconda3/envs/kaggle18/lib/python3.6/site-packages/tensorflow/contrib/image/python/ops/dense_image_warp.py


# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Image warping using per-pixel flow vectors."""


import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


def _interpolate_bilinear(grid,
                          query_points,
                          name='interpolate_bilinear',
                          indexing='ij'):
  """Similar to Matlab's interp2 function.

  Finds values for query points on a grid using bilinear interpolation.

  Args:
    grid: a 4-D float `Tensor` of shape `[batch, height, width, channels]`.
    query_points: a 3-D float `Tensor` of N points with shape `[batch, N, 2]`.
    name: a name for the operation (optional).
    indexing: whether the query points are specified as row and column (ij),
      or Cartesian coordinates (xy).

  Returns:
    values: a 3-D `Tensor` with shape `[batch, N, channels]`

  Raises:
    ValueError: if the indexing mode is invalid, or if the shape of the inputs
      invalid.
  """
  if indexing != 'ij' and indexing != 'xy':
    raise ValueError('Indexing mode must be \'ij\' or \'xy\'')

  with ops.name_scope(name):
    grid = ops.convert_to_tensor(grid)
    query_points = ops.convert_to_tensor(query_points)
    shape = grid.get_shape().as_list()
    if len(shape) != 4:
      msg = 'Grid must be 4 dimensional. Received size: '
      raise ValueError(msg + str(grid.get_shape()))

    batch_size, height, width, channels = shape
    query_type = query_points.dtype
    grid_type = grid.dtype

    if (len(query_points.get_shape()) != 3 or
        query_points.get_shape()[2].value != 2):
      msg = ('Query points must be 3 dimensional and size 2 in dim 2. Received '
             'size: ')
      raise ValueError(msg + str(query_points.get_shape()))

    _, num_queries, _ = query_points.get_shape().as_list()

    if height < 2 or width < 2:
      msg = 'Grid must be at least batch_size x 2 x 2 in size. Received size: '
      raise ValueError(msg + str(grid.get_shape()))

    alphas = []
    floors = []
    ceils = []

    index_order = [0, 1] if indexing == 'ij' else [1, 0]
    unstacked_query_points = array_ops.unstack(query_points, axis=2)

    for dim in index_order:
      with ops.name_scope('dim-' + str(dim)):
        queries = unstacked_query_points[dim]

        size_in_indexing_dimension = shape[dim + 1]

        # max_floor is size_in_indexing_dimension - 2 so that max_floor + 1
        # is still a valid index into the grid.
        max_floor = math_ops.cast(size_in_indexing_dimension - 2, query_type)
        min_floor = constant_op.constant(0.0, dtype=query_type)
        floor = math_ops.minimum(
            math_ops.maximum(min_floor, math_ops.floor(queries)), max_floor)
        int_floor = math_ops.cast(floor, dtypes.int32)
        floors.append(int_floor)
        ceil = int_floor + 1
        ceils.append(ceil)

        # alpha has the same type as the grid, as we will directly use alpha
        # when taking linear combinations of pixel values from the image.
        alpha = math_ops.cast(queries - floor, grid_type)
        min_alpha = constant_op.constant(0.0, dtype=grid_type)
        max_alpha = constant_op.constant(1.0, dtype=grid_type)
        alpha = math_ops.minimum(math_ops.maximum(min_alpha, alpha), max_alpha)

        # Expand alpha to [b, n, 1] so we can use broadcasting
        # (since the alpha values don't depend on the channel).
        alpha = array_ops.expand_dims(alpha, 2)
        alphas.append(alpha)

    if batch_size * height * width > np.iinfo(np.int32).max / 8:
      error_msg = """The image size or batch size is sufficiently large
                     that the linearized addresses used by array_ops.gather
                     may exceed the int32 limit."""
      raise ValueError(error_msg)

    flattened_grid = array_ops.reshape(grid,
                                       [batch_size * height * width, channels])
    batch_offsets = array_ops.reshape(
        math_ops.range(batch_size) * height * width, [batch_size, 1])

    # This wraps array_ops.gather. We reshape the image data such that the
    # batch, y, and x coordinates are pulled into the first dimension.
    # Then we gather. Finally, we reshape the output back. It's possible this
    # code would be made simpler by using array_ops.gather_nd.
    def gather(y_coords, x_coords, name):
      with ops.name_scope('gather-' + name):
        linear_coordinates = batch_offsets + y_coords * width + x_coords
        gathered_values = array_ops.gather(flattened_grid, linear_coordinates)
        return array_ops.reshape(gathered_values,
                                 [batch_size, num_queries, channels])

    # grab the pixel values in the 4 corners around each query point
    top_left = gather(floors[0], floors[1], 'top_left')
    top_right = gather(floors[0], ceils[1], 'top_right')
    bottom_left = gather(ceils[0], floors[1], 'bottom_left')
    bottom_right = gather(ceils[0], ceils[1], 'bottom_right')

    # now, do the actual interpolation
    with ops.name_scope('interpolate'):
      interp_top = alphas[1] * (top_right - top_left) + top_left
      interp_bottom = alphas[1] * (bottom_right - bottom_left) + bottom_left
      interp = alphas[0] * (interp_bottom - interp_top) + interp_top

    return interp


def dense_image_warp(image, flow, name='dense_image_warp'):
  """Image warping using per-pixel flow vectors.

  Apply a non-linear warp to the image, where the warp is specified by a dense
  flow field of offset vectors that define the correspondences of pixel values
  in the output image back to locations in the  source image. Specifically, the
  pixel value at output[b, j, i, c] is
  images[b, j - flow[b, j, i, 0], i - flow[b, j, i, 1], c].

  The locations specified by this formula do not necessarily map to an int
  index. Therefore, the pixel value is obtained by bilinear
  interpolation of the 4 nearest pixels around
  (b, j - flow[b, j, i, 0], i - flow[b, j, i, 1]). For locations outside
  of the image, we use the nearest pixel values at the image boundary.


  Args:
    image: 4-D float `Tensor` with shape `[batch, height, width, channels]`.
    flow: A 4-D float `Tensor` with shape `[batch, height, width, 2]`.
    name: A name for the operation (optional).

    Note that image and flow can be of type tf.half, tf.float32, or tf.float64,
    and do not necessarily have to be the same type.

  Returns:
    A 4-D float `Tensor` with shape`[batch, height, width, channels]`
      and same type as input image.

  Raises:
    ValueError: if height < 2 or width < 2 or the inputs have the wrong number
                of dimensions.
  """
  with ops.name_scope(name):
    batch_size, height, width, channels = image.get_shape().as_list()
    # The flow is defined on the image grid. Turn the flow into a list of query
    # points in the grid space.
    grid_x, grid_y = array_ops.meshgrid(
        math_ops.range(width), math_ops.range(height))
    stacked_grid = math_ops.cast(
        array_ops.stack([grid_y, grid_x], axis=2), flow.dtype)
    batched_grid = array_ops.expand_dims(stacked_grid, axis=0)
    query_points_on_grid = batched_grid - flow
    query_points_flattened = array_ops.reshape(query_points_on_grid,
                                               [batch_size, height * width, 2])
    # Compute values at the query points, then reshape the result back to the
    # image grid.
    interpolated = _interpolate_bilinear(image, query_points_flattened)
    interpolated = array_ops.reshape(interpolated,
                                     [batch_size, height, width, channels])
    return interpolated





# /Users/edufonseca/miniconda3/envs/kaggle18/lib/python3.6/site-packages/tensorflow/contrib/image/python/ops/sparse_image_warp.py
# /Users/edufonseca/miniconda3/envs/kaggle18/lib/python3.6/site-packages/tensorflow/contrib/image/python/ops/sparse_image_warp.py
# /Users/edufonseca/miniconda3/envs/kaggle18/lib/python3.6/site-packages/tensorflow/contrib/image/python/ops/sparse_image_warp.py
"""Image warping using sparse flow defined at control points."""
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import numpy as np

# from tensorflow.contrib.image.python.ops import dense_image_warp
# from tensorflow.contrib.image.python.ops import interpolate_spline

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops


def _get_grid_locations(image_height, image_width):
  """Wrapper for np.meshgrid."""

  y_range = np.linspace(0, image_height - 1, image_height)
  x_range = np.linspace(0, image_width - 1, image_width)
  y_grid, x_grid = np.meshgrid(y_range, x_range, indexing='ij')
  return np.stack((y_grid, x_grid), -1)


def _expand_to_minibatch(np_array, batch_size):
  """Tile arbitrarily-sized np_array to include new batch dimension."""
  tiles = [batch_size] + [1] * np_array.ndim
  return np.tile(np.expand_dims(np_array, 0), tiles)


def _get_boundary_locations(image_height, image_width, num_points_per_edge):
  """Compute evenly-spaced indices along edge of image."""
  y_range = np.linspace(0, image_height - 1, num_points_per_edge + 2)
  x_range = np.linspace(0, image_width - 1, num_points_per_edge + 2)
  ys, xs = np.meshgrid(y_range, x_range, indexing='ij')
  is_boundary = np.logical_or(
      np.logical_or(xs == 0, xs == image_width - 1),
      np.logical_or(ys == 0, ys == image_height - 1))
  return np.stack([ys[is_boundary], xs[is_boundary]], axis=-1)


def _add_zero_flow_controls_at_boundary(control_point_locations,
                                        control_point_flows, image_height,
                                        image_width, boundary_points_per_edge):
  """Add control points for zero-flow boundary conditions.

   Augment the set of control points with extra points on the
   boundary of the image that have zero flow.

  Args:
    control_point_locations: input control points
    control_point_flows: their flows
    image_height: image height
    image_width: image width
    boundary_points_per_edge: number of points to add in the middle of each
                           edge (not including the corners).
                           The total number of points added is
                           4 + 4*(boundary_points_per_edge).

  Returns:
    merged_control_point_locations: augmented set of control point locations
    merged_control_point_flows: augmented set of control point flows
  """

  batch_size = control_point_locations.get_shape()[0].value

  boundary_point_locations = _get_boundary_locations(image_height, image_width,
                                                     boundary_points_per_edge)

  boundary_point_flows = np.zeros([boundary_point_locations.shape[0], 2])

  type_to_use = control_point_locations.dtype
  boundary_point_locations = constant_op.constant(
      _expand_to_minibatch(boundary_point_locations, batch_size),
      dtype=type_to_use)

  boundary_point_flows = constant_op.constant(
      _expand_to_minibatch(boundary_point_flows, batch_size), dtype=type_to_use)

  merged_control_point_locations = array_ops.concat(
      [control_point_locations, boundary_point_locations], 1)

  merged_control_point_flows = array_ops.concat(
      [control_point_flows, boundary_point_flows], 1)

  return merged_control_point_locations, merged_control_point_flows


def sparse_image_warp(image,
                      source_control_point_locations,
                      dest_control_point_locations,
                      interpolation_order=2,
                      regularization_weight=0.0,
                      num_boundary_points=0,
                      name='sparse_image_warp'):
  """Image warping using correspondences between sparse control points.

  Apply a non-linear warp to the image, where the warp is specified by
  the source and destination locations of a (potentially small) number of
  control points. First, we use a polyharmonic spline
  (@{tf.contrib.image.interpolate_spline}) to interpolate the displacements
  between the corresponding control points to a dense flow field.
  Then, we warp the image using this dense flow field
  (@{tf.contrib.image.dense_image_warp}).

  Let t index our control points. For regularization_weight=0, we have:
  warped_image[b, dest_control_point_locations[b, t, 0],
                  dest_control_point_locations[b, t, 1], :] =
  image[b, source_control_point_locations[b, t, 0],
           source_control_point_locations[b, t, 1], :].

  For regularization_weight > 0, this condition is met approximately, since
  regularized interpolation trades off smoothness of the interpolant vs.
  reconstruction of the interpolant at the control points.
  See @{tf.contrib.image.interpolate_spline} for further documentation of the
  interpolation_order and regularization_weight arguments.


  Args:
    image: `[batch, height, width, channels]` float `Tensor`
    source_control_point_locations: `[batch, num_control_points, 2]` float
      `Tensor`
    dest_control_point_locations: `[batch, num_control_points, 2]` float
      `Tensor`
    interpolation_order: polynomial order used by the spline interpolation
    regularization_weight: weight on smoothness regularizer in interpolation
    num_boundary_points: How many zero-flow boundary points to include at
      each image edge.Usage:
        num_boundary_points=0: don't add zero-flow points
        num_boundary_points=1: 4 corners of the image
        num_boundary_points=2: 4 corners and one in the middle of each edge
          (8 points total)
        num_boundary_points=n: 4 corners and n-1 along each edge
    name: A name for the operation (optional).

    Note that image and offsets can be of type tf.half, tf.float32, or
    tf.float64, and do not necessarily have to be the same type.

  Returns:
    warped_image: `[batch, height, width, channels]` float `Tensor` with same
      type as input image.
    flow_field: `[batch, height, width, 2]` float `Tensor` containing the dense
      flow field produced by the interpolation.
  """

  image = ops.convert_to_tensor(image)
  source_control_point_locations = ops.convert_to_tensor(
      source_control_point_locations)
  dest_control_point_locations = ops.convert_to_tensor(
      dest_control_point_locations)

  control_point_flows = (
      dest_control_point_locations - source_control_point_locations)

  clamp_boundaries = num_boundary_points > 0
  boundary_points_per_edge = num_boundary_points - 1

  with ops.name_scope(name):

    batch_size, image_height, image_width, _ = image.get_shape().as_list()

    # This generates the dense locations where the interpolant
    # will be evaluated.
    grid_locations = _get_grid_locations(image_height, image_width)

    flattened_grid_locations = np.reshape(grid_locations,
                                          [image_height * image_width, 2])

    flattened_grid_locations = constant_op.constant(
        _expand_to_minibatch(flattened_grid_locations, batch_size), image.dtype)

    if clamp_boundaries:
      (dest_control_point_locations,
       control_point_flows) = _add_zero_flow_controls_at_boundary(
           dest_control_point_locations, control_point_flows, image_height,
           image_width, boundary_points_per_edge)

    # flattened_flows = interpolate_spline.interpolate_spline(
    flattened_flows=interpolate_spline(
            dest_control_point_locations, control_point_flows,
        flattened_grid_locations, interpolation_order, regularization_weight)

    dense_flows = array_ops.reshape(flattened_flows,
                                    [batch_size, image_height, image_width, 2])

    # warped_image = dense_image_warp.dense_image_warp(image, dense_flows)
    warped_image = dense_image_warp(image, dense_flows)

    return warped_image, dense_flows






























# ---------------------------

def spec_augment(mel_spectrogram, time_warping_para=80, frequency_masking_para=27,
                 time_masking_para=100, frequency_mask_num=1, time_mask_num=1):
    """Spec augmentation Calculation Function.
    'SpecAugment' have 3 steps for audio data augmentation.
    first step is time warping using Tensorflow's image_sparse_warp function.
    Second step is frequency masking, last step is time masking.
    # Arguments:
      mel_spectrogram(numpy array): audio file path of you want to warping and masking.
      time_warping_para(float): Augmentation parameter, "time warp parameter W".
        If none, default = 80 for LibriSpeech.
      frequency_masking_para(float): Augmentation parameter, "frequency mask parameter F"
        If none, default = 100 for LibriSpeech.
      time_masking_para(float): Augmentation parameter, "time mask parameter T"
        If none, default = 27 for LibriSpeech.
      frequency_mask_num(float): number of frequency masking lines, "m_F".
        If none, default = 1 for LibriSpeech.
      time_mask_num(float): number of time masking lines, "m_T".
        If none, default = 1 for LibriSpeech.
    # Returns
      mel_spectrogram(numpy array): warped and masked mel spectrogram.
    """
    # v is mel bands / tau is time frames
    v = mel_spectrogram.shape[0]
    tau = mel_spectrogram.shape[1]

    # Step 1 : Time warping
    # Image warping control point setting.
    mel_spectrogram_holder = tf.placeholder(tf.float32, shape=[1, v, tau, 1])
    location_holder = tf.placeholder(tf.float32, shape=[1, 1, 2])
    destination_holder = tf.placeholder(tf.float32, shape=[1, 1, 2])

    center_position = v/2
    random_point = np.random.randint(low=time_warping_para, high=tau - time_warping_para)
    # warping distance chose.
    w = np.random.uniform(low=0, high=time_warping_para)

    control_point_locations = [[center_position, random_point]]
    control_point_locations = np.float32(np.expand_dims(control_point_locations, 0))

    control_point_destination = [[center_position, random_point + w]]
    control_point_destination = np.float32(np.expand_dims(control_point_destination, 0))

    # mel spectrogram data type convert to tensor constant for sparse_image_warp.
    mel_spectrogram = mel_spectrogram.reshape([1, mel_spectrogram.shape[0], mel_spectrogram.shape[1], 1])
    mel_spectrogram = np.float32(mel_spectrogram)

    warped_mel_spectrogram_op, _ = sparse_image_warp(mel_spectrogram_holder,
                                                     source_control_point_locations=location_holder,
                                                     dest_control_point_locations=destination_holder,
                                                     interpolation_order=2,
                                                     regularization_weight=0,
                                                     num_boundary_points=1
                                                     )

    # Change warp result's data type to numpy array for masking step.
    feed_dict = {mel_spectrogram_holder:mel_spectrogram,
                 location_holder:control_point_locations,
                 destination_holder:control_point_destination}

    with tf.Session() as sess:
        warped_mel_spectrogram = sess.run(warped_mel_spectrogram_op, feed_dict=feed_dict)

    warped_mel_spectrogram = warped_mel_spectrogram.reshape([warped_mel_spectrogram.shape[1],
                                                             warped_mel_spectrogram.shape[2]])

    # Step 2 : Frequency masking
    for i in range(frequency_mask_num):
        f = np.random.uniform(low=0.0, high=frequency_masking_para)
        f = int(f)
        f0 = random.randint(0, v - f)
        warped_mel_spectrogram[f0:f0 + f, :] = 0

    # Step 3 : Time masking
    for i in range(time_mask_num):
        t = np.random.uniform(low=0.0, high=time_masking_para)
        t = int(t)
        t0 = random.randint(0, tau - t)
        warped_mel_spectrogram[:, t0:t0 + t] = 0

    return warped_mel_spectrogram


def visualization_spectrogram(mel_spectrogram, title):
    """visualizing result of SpecAugment
    # Arguments:
      mel_spectrogram(ndarray): mel_spectrogram to visualize.
      title(String): plot figure's title
    """
    # Show mel-spectrogram using librosa's specshow.
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(mel_spectrogram, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
    # plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()