import pytest
import copy
import numpy as np
from environment import Env, StaticUpFlow



testdata0 = [
    (0., 1., 0.5, [ # center = 0, left boarder = -0.5, right boarder = 0.5
        np.array([-7., 9.], dtype=float), np.array([-0.5, 2.], dtype=float),
        np.array([0., 0.], dtype=float),np.array([0.5, -1.], dtype=float),
        np.array([10., -4.], dtype=float)
        ],
        [ # mid_value = 0.5 (boarders & outside = 0.)
        np.array([0., 0.], dtype=float), np.array([0., 0.], dtype=float),
        np.array([0., 0.5], dtype=float), np.array([0., 0.], dtype=float),
        np.array([0., 0.], dtype=float)
        ]),
    (-5., 2., 1., [ # center = -5, left boarder = -6., right boarder = -4.
        np.array([-12., 7.], dtype=float), np.array([-6., 8.], dtype=float),
        np.array([-5., -2.], dtype=float), np.array([-4., 3.], dtype=float),
        np.array([2., 0.], dtype=float)
        ],
        [ # mid_value = 1. (boarders & outside = 0.)
        np.array([0., 0.], dtype=float), np.array([0., 0.], dtype=float),
        np.array([0., 1.], dtype=float), np.array([0., 0.], dtype=float),
        np.array([0., 0.], dtype=float)
        ]),
    (2., 4., 2., [ # center = 2. left boarder = 0., right boarder = 4.
        np.array([-1., 9.], dtype=float), np.array([0., 5.], dtype=float),
        np.array([2., -3.], dtype=float), np.array([4., 2.], dtype=float),
        np.array([7., -1.], dtype=float)
        ],
        [ # mid_value = 2. (boarders & outside = 0.)
        np.array([0., 0.], dtype=float), np.array([0., 0.], dtype=float),
        np.array([0., 2.], dtype=float), np.array([0., 0.], dtype=float),
        np.array([0., 0.], dtype=float)
        ])
    ]

@pytest.mark.parametrize("center,width,mid_value,points,expected", testdata0)
def test_staticupflow(center, width, mid_value, points, expected):
    upflow = StaticUpFlow(center, width, mid_value)
    for t in range(10):
        vectors = [upflow(point, t) for point in points]
        
        assert np.allclose(vectors, expected)

testdata1 = [
    (np.array([[-1., 0.,], [0., 5.], [1., 2.]], dtype=float),
    np.array([[0., 0.,], [0., 1.], [0., 0.]], dtype=float)),
    (np.array([[9., -1.], [1., 5.], [7., 8.]], dtype=float),
    np.array([[0., 0.,], [0., 0.], [0., 0.]], dtype=float)),
    (np.array([[0., -9.], [0., 2.], [0., 1.]], dtype=float),
    np.array([[0., 1.,], [0., 1.], [0., 1.]], dtype=float))
    ]

@pytest.mark.parametrize("points,expected", testdata1)
def test_env_with_staticupflow(points, expected):
    upflow = StaticUpFlow(0., 2., 1.)
    env = Env(upflow, 0.01)
    vectors = env.evaluate(points)

    assert np.allclose(vectors, expected)
