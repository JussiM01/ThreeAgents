import numpy as np
import pytest
from utils import rotate_all


sq = 1/np.sqrt(2)
testdata0 = [
    (np.array([[0., 1.], [1., 0.], [0., -1.]], dtype=float), np.pi,
     np.array([[0., -1.], [-1., 0.], [0., 1.]], dtype=float)),
    (np.array([[0., 1.], [1., 0.], [0., -1.]], dtype=float), -np.pi,
     np.array([[0., -1.], [-1., 0.], [0., 1.]], dtype=float)),
    (np.array([[0., 1.], [1., 0.], [0., -1.]], dtype=float), np.pi/2,
     np.array([[-1., 0.], [0., 1.], [1., 0.]], dtype=float)),
    (np.array([[0., 1.], [1., 0.], [0., -1.]], dtype=float), -np.pi/2,
     np.array([[1., 0.], [0., -1.], [-1., 0.]], dtype=float)),
    (np.array([[0., 1.], [1., 0.], [0., -1.]], dtype=float), np.pi/4,
     np.array([[-sq, sq], [sq, sq], [sq, -sq]], dtype=float)),
    (np.array([[0., 1.], [1., 0.], [0., -1.]], dtype=float), -np.pi/4,
     np.array([[sq, sq], [sq, -sq], [-sq, -sq]], dtype=float))
    ]


@pytest.mark.parametrize("vectors,angle,expected", testdata0)
def test_rotate_all(vectors, angle, expected):
    points = np.array([[0., 1.], [2., 3.], [4., 5.]])
    new_vectors = rotate_all(vectors, angle)

    assert np.allclose(new_vectors, expected)
