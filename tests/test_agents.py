import pytest
import copy
import numpy as np
from agents import MultiAgent



testdata0 = [
    (np.array([[-0.5, 0.], [0.5, 0.], [0., -0.5]], dtype=float), 0.5, 1.),
    (np.array([[-1., 0.], [1., 0.], [0., 1.]], dtype=float), 1., 1.),
    (np.array([[2., -2.], [-2., 2.], [2., 2.]], dtype=float), 2., 1.),
    (np.array([[4., -2.], [-2., 4.], [-2., -2]], dtype=float), 3., 1.),
    (np.array([[-0.1, 0.], [0.1, 0.], [0., -0.1]], dtype=float), 0.5, -1.),
    (np.array([[-0.3, 0.], [0.3, 0.], [0., 0.3]], dtype=float), 1., -1.),
    (np.array([[0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]], dtype=float), 2., -1.),
    (np.array([[1.5, -0.5], [-0.5, 1.5], [-0.5, -0.5]], dtype=float), 3., -1.)
    ]

@pytest.mark.parametrize("points,target_distance,expected_sign", testdata0)
def test_sign_reshape_step(points, target_distance, expected_sign):

    init_points = copy.deepcopy(points)

    model = MultiAgent(init_points, target_distance, 1., 100., 0.05, 0.001)
    model._reshape_step('triangle', 1.)

    old_distances = [
        np.linalg.norm(points[0] - points[1]),
        np.linalg.norm(points[0] - points[2]),
        np.linalg.norm(points[1] - points[2])
        ]

    new_distances = [
        np.linalg.norm(model.positions[0] - model.positions[1]),
        np.linalg.norm(model.positions[0] - model.positions[2]),
        np.linalg.norm(model.positions[1] - model.positions[2])
        ]

    old_distances = np.stack(old_distances, axis=0)
    new_distances = np.stack(new_distances, axis=0)

    differences = old_distances - new_distances

    for i in range(3):
        assert differences[i]*expected_sign > 0



testdata1 = [
    (np.array([[-1. + 9e-7, 0.], [1., 0.], [0., np.sqrt(3)]],
        dtype=float), 1e-6),
    (np.array([[-1., 0.], [1., 0. + 9e-6], [0., np.sqrt(3)]],
        dtype=float), 1e-5),
    (np.array([[-1., 0.], [1., 0.], [0. + 9e-5, np.sqrt(3)]],
        dtype=float), 1e-4),
    (np.array([[-1., 0. + 3e-4], [1. + 3e-4, 0.], [0., np.sqrt(3) + 3e-4]],
        dtype=float), 1e-3)
    ]

@pytest.mark.parametrize("points,accepted_error", testdata1)
def test_stopping_reshape_formation(points, accepted_error):

    init_points = copy.deepcopy(points)

    model = MultiAgent(init_points, 2., 1., 100., 0.05, accepted_error)
    model.reshape_formation('triangle', 1.)

    new_positions = model.positions

    assert np.array_equal(points, new_positions)


testdata2 = [
    (np.array([[-1., 0.], [1., 0.], [0., np.sqrt(3)]], dtype=float),
        0.005, np.array([1., 0.], dtype=float), 10.,
        np.array([[-0.95, 0.], [1.05, 0.], [0.05, np.sqrt(3)]], dtype=float)),
    (np.array([[-1., 0.], [1., 0.], [0., np.sqrt(3)]], dtype=float),
        0.01, np.array([0., 1.], dtype=float), 20.,
        np.array([[-1, 0.2], [1., 0.2], [0., np.sqrt(3)+0.2]], dtype=float)),
    (np.array([[-1., 0.], [1., 0.], [0., np.sqrt(3)]], dtype=float),
        0.02, np.array([0.5, 0.5], dtype=float), 50.,
        np.array([[0., 1.], [2., 1.], [1., np.sqrt(3)+1.]], dtype=float)),
    (np.array([[-1., 0.], [1., 0.], [0., np.sqrt(3)]], dtype=float),
        0.02, np.array([0.5, 0.5], dtype=float), 100., # speed will be cliped
        (np.array([[0., 1.], [2., 1.], [1., np.sqrt(3)+1.]], dtype=float))
    ]

@pytest.mark.parametrize("points,delta,direction,velocity,expected", testdata2)
def test_shift_step("points,delta,direction,velocity,expected"):

    init_points = copy.deepcopy(points)

    model = MultiAgent(init_points, 1., 1., 50., delta, 0.001) # max speed = 50.
    model._shift_step(direction, velocity)

    new_positions = model.positions

    assert new_positions == expected
#
#
# testdata3 = ...
#
# @pytest.mark.parametrize("points,delta,target,velocity", testdata3)
# def test_stopping_shift_formation("points,delta,target,velocity"):
#
#     init_points = copy.deepcopy(points)
#
#     model = MultiAgent(init_points, 1., 1., 50., delta, 0.001)
#     model.shift_formation(target, velocity)
#
#     new_positions = model.positions
#
#     assert np.array_equal(points, new_positions)
#
#
# testdata4 = ...
#
# @pytest.mark.parametrize("points,delta,target,velocity", testdata4)
# def test_over_shooting_prevention("points,delta,target,velocity"):
#
#     init_points = copy.deepcopy(points)
#
#     model = MultiAgent(init_points, 1., 1., 50., delta, 0.001)
#     model.shift_formation(target, velocity)
#
#     new_positions = model.positions
#
#     assert np.array_equal(points, new_positions)
