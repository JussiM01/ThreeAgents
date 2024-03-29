import numpy as np
import pytest

from copy import deepcopy
from interactionmodels import CentralControl
from utils import normalize, normalize_all


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
    init_points = deepcopy(points)
    model = CentralControl(init_points, target_distance, 1., 100., 0.05, 0.001)
    model._reshape_step(1.)
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
    (np.array([[-1. + 9e-7, 0.], [1., 0.], [0., np.sqrt(3)]], dtype=float),
     1e-6),
    (np.array([[-1., 0.], [1., 0. + 9e-6], [0., np.sqrt(3)]], dtype=float),
     1e-5),
    (np.array([[-1., 0.], [1., 0.], [0. + 9e-5, np.sqrt(3)]], dtype=float),
     1e-4),
    (np.array([[-1., 0. + 3e-4], [1. + 3e-4, 0.], [0., np.sqrt(3) + 3e-4]],
     dtype=float), 1e-3)
    ]


@pytest.mark.parametrize("points,accepted_error", testdata1)
def test_stopping_reshape_formation(points, accepted_error):
    init_points = deepcopy(points)
    model = CentralControl(init_points, 2., 1., 100., 0.05, accepted_error)
    model.reshape_formation(1.)
    new_positions = model.positions

    assert np.array_equal(points, new_positions)


testdata2 = [
    (np.array([[-1., 0.], [1., 0.], [0., np.sqrt(3)]], dtype=float),
        0.005, np.array([1., 0.], dtype=float), 10.,
        np.array([[-0.95, 0.], [1.05, 0.], [0.05, np.sqrt(3)]], dtype=float)),
    (np.array([[-1., 0.], [1., 0.], [0., np.sqrt(3)]], dtype=float),
        0.01, np.array([0., 1.], dtype=float), 20.,
        np.array([[-1, 0.2], [1., 0.2], [0., np.sqrt(3) + 0.2]], dtype=float)),
    (np.array([[-1., 0.], [1., 0.], [0., np.sqrt(3)]], dtype=float),
        0.02, np.array([0.5, 0.5], dtype=float), 50.,
        np.array([[-0.5, 0.5], [1.5, 0.5], [0.5, np.sqrt(3)+0.5]],
                 dtype=float)),
    (np.array([[-1., 0.], [1., 0.], [0., np.sqrt(3)]], dtype=float),
        0.02, np.array([-1., 0.], dtype=float), 100.,  # speed will be cliped
        np.array([[-2., 0.], [0., 0.], [-1., np.sqrt(3)]], dtype=float)),
    (np.array([[-1., 0.], [1., 0.], [0., np.sqrt(3)]], dtype=float),
        0.02, np.array([0., -1.], dtype=float), 100.,  # speed will be cliped
        np.array([[-1., -1.], [1., -1.], [0., np.sqrt(3) - 1.]], dtype=float)),
    (np.array([[0., -1.], [0., 1.], [np.sqrt(3), 0.]], dtype=float),
        0.02, np.array([0., 1.], dtype=float), 100.,  # speed will be cliped
        np.array([[0., 0.], [0., 2.], [np.sqrt(3), 1.]], dtype=float))
    ]


@pytest.mark.parametrize("points,delta,direction,speed,expected", testdata2)
def test_shift_step(points, delta, direction, speed, expected):
    init_points = deepcopy(points)
    model = CentralControl(init_points, 1., 1., 50., delta, 0.001)
    model._shift_step(direction, speed)
    new_positions = model.positions

    assert np.array_equal(new_positions, expected)


testdata3 = [
    (np.array([[-1.+1e-6, 0.], [1.+1e-6, 0.], [1e-6, np.sqrt(3)]],
              dtype=float), [0., np.sqrt(3)/3], 1e-5),
    (np.array([[-1., 1e-6], [1., 1e-6], [0., np.sqrt(3)+1e-6]], dtype=float),
        [0., np.sqrt(3)/3], 1e-5),
    (np.array([[0., -1.+1e-5], [0., 1.+1e-5], [np.sqrt(3), 1e-5]],
              dtype=float), [np.sqrt(3)/3, 0.], 1e-4),
    (np.array([[1e-5, -1.], [1e-5, 1.], [np.sqrt(3)+1e-5, 0.]], dtype=float),
        [np.sqrt(3)/3, 0.], 1e-4),
    (np.array([[-1.+1e-4, 2.], [1.+1e-4, 2.], [1e-4, 2.+np.sqrt(3)]],
              dtype=float), [0., 2+np.sqrt(3)/3], 1e-3),
    (np.array([[-5., 1e-4], [-3., 1e-4], [-4., np.sqrt(3)+1e-4]], dtype=float),
        [-4., np.sqrt(3)/3], 1e-3),
    ]


@pytest.mark.parametrize("points,target,error,", testdata3)
def test_stopping_shift_formation(points, target, error):
    init_points = deepcopy(points)
    model = CentralControl(init_points, 1., 1., 50., 0.01, error)
    model.shift_formation(target, 10.)
    new_positions = model.positions

    assert np.array_equal(points, new_positions)


testdata4 = [
    (np.array([[-1.+1e-6, 0.], [1.+1e-6, 0.], [1e-6, np.sqrt(3)]],
              dtype=float), [0., np.sqrt(3)/3], 5e-6, 1e-7),
    (np.array([[-1., 1e-6], [1., 1e-6], [0., np.sqrt(3)+1e-6]], dtype=float),
        [0., np.sqrt(3)/3], 5e-6, 1e-7),
    (np.array([[0., -1.+1e-5], [0., 1.+1e-5], [np.sqrt(3), 1e-5]],
              dtype=float), [np.sqrt(3)/3, 0.], 5e-5, 1e-7),
    (np.array([[1e-5, -1.], [1e-5, 1.], [np.sqrt(3)+1e-5, 0.]], dtype=float),
        [np.sqrt(3)/3, 0.], 5e-5, 1e-7),
    (np.array([[-1.+1e-4, 2.], [1.+1e-4, 2.], [1e-4, 2.+np.sqrt(3)]],
              dtype=float), [0., 2+np.sqrt(3)/3], 5e-4, 1e-7),
    (np.array([[-5., 1e-4], [-3., 1e-4], [-4., np.sqrt(3)+1e-4]], dtype=float),
        [-4., np.sqrt(3)/3], 5e-5, 1e-7),
    ]


@pytest.mark.parametrize("points,target,delta,error", testdata4)
def test_over_shooting_prevention(points, target, delta, error):
    init_points = deepcopy(points)
    model = CentralControl(init_points, 1., 1., 50., delta, error)
    model.shift_formation(target, 20.)
    new_mean = np.mean(model.positions, axis=0)

    assert np.allclose(new_mean, target)


testdata5 = [
    (np.array([[np.cos(np.pi/4), np.sin(np.pi/4)],
     [np.cos(np.pi/4 + 2*np.pi/3), np.sin(np.pi/4 + 2*np.pi/3)],
     [np.cos(np.pi/4 + 4*np.pi/3), np.sin(np.pi/4 + 4*np.pi/3)]], dtype=float),
     0.1, -1, np.pi/8, 10*np.pi/8,
     np.array([np.cos(np.pi/8), np.sin(np.pi/8)], dtype=float)),
    (np.array([[np.cos(np.pi/4), np.sin(np.pi/4)],
     [np.cos(np.pi/4 + 2*np.pi/3), np.sin(np.pi/4 + 2*np.pi/3)],
     [np.cos(np.pi/4 + 4*np.pi/3), np.sin(np.pi/4 + 4*np.pi/3)]], dtype=float),
     0.05, 1, np.pi/8, 20*np.pi/8,
     np.array([np.cos(3*np.pi/8), np.sin(3*np.pi/8)], dtype=float)),
    (np.array([[np.cos(np.pi/2), np.sin(np.pi/2)],
     [np.cos(np.pi/2 + 2*np.pi/3), np.sin(np.pi/2 + 2*np.pi/3)],
     [np.cos(np.pi/2 + 4*np.pi/3), np.sin(np.pi/2 + 4*np.pi/3)]], dtype=float),
     0.01, -1, np.pi/8, 100*np.pi/8,
     np.array([np.cos(3*np.pi/8), np.sin(3*np.pi/8)], dtype=float)),
    (np.array([[np.cos(-np.pi/2), np.sin(-np.pi/2)],
     [np.cos(-np.pi/2 + 2*np.pi/3), np.sin(-np.pi/2 + 2*np.pi/3)],
     [np.cos(-np.pi/2 + 4*np.pi/3), np.sin(-np.pi/2 + 4*np.pi/3)]],
             dtype=float), 0.005, 1, np.pi/8, 200*np.pi/8,
     np.array([np.cos(-3*np.pi/8), np.sin(-3*np.pi/8)], dtype=float))
            ]


@pytest.mark.parametrize("points,delta,sign,angle,speed,expected", testdata5)
def test_turn_step(points, delta, sign, angle, speed, expected):
    init_points = deepcopy(points)
    model = CentralControl(init_points, 1., 1., 1000., delta, 0.001)
    model.task_params['rotation_sign'] = sign
    model.task_params['rotation_center'] = np.array([0., 0.], dtype=float)
    model._turn_step(angle, speed)
    new_lead_position = model.positions[0, :]

    assert np.allclose(new_lead_position, expected)


testdata6 = [
    (np.array([[np.cos(1e-6), np.sin(1e-6)],
     [np.cos(1e-6 + 2*np.pi/3), np.sin(1e-6 + 2*np.pi/3)],
     [np.cos(1e-6 + 4*np.pi/3), np.sin(1e-6 + 4*np.pi/3)]], dtype=float),
     [10., 0.], 10., 1e-5),
    (np.array([[np.cos(np.pi/2 - 1e-5), np.sin(np.pi/2 - 1e-5)],
     [np.cos(np.pi/2 - 1e-5 + 2*np.pi/3), np.sin(np.pi/2 - 1e-5 + 2*np.pi/3)],
     [np.cos(np.pi/2 - 1e-5 + 4*np.pi/3), np.sin(np.pi/2 - 1e-5 + 4*np.pi/3)]],
     dtype=float), [0., 20.], 20., 1e-4),
    (np.array([[np.cos(np.pi/4 - 1e-4), np.sin(np.pi/4 - 1e-4)],
     [np.cos(np.pi/4 - 1e-4 + 2*np.pi/3), np.sin(np.pi/4 - 1e-4 + 2*np.pi/3)],
     [np.cos(np.pi/4 - 1e-4 + 4*np.pi/3), np.sin(np.pi/4 - 1e-4 + 4*np.pi/3)]],
     dtype=float), [30., 30.], 50., 1e-3),
    (np.array([[np.cos(-np.pi/4 - 1e-3), np.sin(-np.pi/4 - 1e-3)],
     [np.cos(-np.pi/4 - 1e-3 + 2*np.pi/3),
      np.sin(-np.pi/4 - 1e-3 + 2*np.pi/3)],
     [np.cos(-np.pi/4 - 1e-3 + 4*np.pi/3),
      np.sin(-np.pi/4 - 1e-3 + 4*np.pi/3)]], dtype=float), [50., -50.], 100.,
     1e-2),
            ]


@pytest.mark.parametrize("points,target,speed,error,", testdata6)
def test_stopping_turn_formation(points, target, speed, error):
    init_points = deepcopy(points)
    model = CentralControl(init_points, 1., 1., 50., 0.01, error)
    model.turn_formation(target, speed)
    new_positions = model.positions

    assert np.array_equal(points, new_positions)


testdata7 = [
    (np.array([[np.cos(np.pi/4 + 1e-8), np.sin(np.pi/4 + 1e-8)],
     [np.cos(np.pi/4 + 2*np.pi/3 + 1e-8), np.sin(np.pi/4 + 2*np.pi/3 + 1e-8)],
     [np.cos(np.pi/4 + 4*np.pi/3 + 1e-8), np.sin(np.pi/4 + 4*np.pi/3 + 1e-8)]],
     dtype=float), [10., 10.], 0.001, 1e-7),
    (np.array([[np.cos(np.pi/4 - 1e-7), np.sin(np.pi/4 - 1e-7)],
     [np.cos(np.pi/4 + 2*np.pi/3 - 1e-7), np.sin(np.pi/4 + 2*np.pi/3 - 1e-7)],
     [np.cos(np.pi/4 + 4*np.pi/3 - 1e-7), np.sin(np.pi/4 + 4*np.pi/3 - 1e-7)]],
     dtype=float), [20., 20.], 0.002, 1e-6),
    (np.array([[np.cos(np.pi/2 + 1e-6), np.sin(np.pi/2 + 1e-6)],
     [np.cos(np.pi/2 + 2*np.pi/3 + 1e-6), np.sin(np.pi/2 + 2*np.pi/3 + 1e-6)],
     [np.cos(np.pi/2 + 4*np.pi/3 + 1e-6), np.sin(np.pi/2 + 4*np.pi/3 + 1e-6)]],
     dtype=float), [0., 50.], 0.01, 1e-5),
    (np.array([[np.cos(-np.pi/2 + 1e-5), np.sin(-np.pi/2 + 1e-5)],
     [np.cos(-np.pi/2 + 2*np.pi/3 + 1e-5),
      np.sin(-np.pi/2 + 2*np.pi/3 + 1e-5)],
     [np.cos(-np.pi/2 + 4*np.pi/3 + 1e-5),
      np.sin(-np.pi/2 + 4*np.pi/3 + 1e-5)]],
     dtype=float), [-100., -100], 0.05, 1e-4)
            ]


@pytest.mark.parametrize("points,target,delta,error", testdata7)
def test_over_turning_prevention(points, target, delta, error):
    init_points = deepcopy(points)
    rotation_center = np.mean(points)
    model = CentralControl(init_points, 1., 1., 1000., delta, error)
    model.turn_formation(target, 100.)
    target_direction = normalize(target - rotation_center)
    center_to_points = model.positions - rotation_center
    directions = normalize_all(center_to_points)
    differences = np.linalg.norm(directions - target_direction, axis=1)

    assert np.min(differences) < error
