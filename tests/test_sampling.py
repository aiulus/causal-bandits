import numpy as np
import matplotlib.pyplot as plt
from src.utils import SCM
def test_scm_sampling():
    # Create a simple SCM for testing
    scm_input = {
        "nodes": ["X", "Y"],
        "edges": [["X", "Y"]],
        "functions": {
            "Y": "lambda X: 2*X + 1"
        },
        "noise": {
            "X": "N(0,1)",
            "Y": "N(0,1)"
        }
    }

    scm = SCM(scm_input)
    n_samples = 10000
    samples = scm.sample(n_samples)

    X_samples = samples["X"]
    Y_samples = samples["Y"]

    # Joint distribution P(X, Y)
    joint_hist, x_edges, y_edges = np.histogram2d(X_samples, Y_samples, bins=50, density=True)

    # Marginal distribution P(X)
    x_hist, _ = np.histogram(X_samples, bins=x_edges, density=True)

    # Marginal distribution P(Y)
    y_hist, _ = np.histogram(Y_samples, bins=y_edges, density=True)

    # Conditional distribution P(Y|X)
    cond_y_given_x = joint_hist / x_hist[:, None]

    # Conditional distribution P(X|Y)
    cond_x_given_y = joint_hist / y_hist[None, :]

    # Reconstruct joint distribution from P(Y|X) * P(X)
    joint_reconstructed_y_given_x = cond_y_given_x * x_hist[:, None]

    # Reconstruct joint distribution from P(X|Y) * P(Y)
    joint_reconstructed_x_given_y = cond_x_given_y * y_hist[None, :]

    # Compare the reconstructed joint distributions with the original joint distribution
    assert np.allclose(joint_hist, joint_reconstructed_y_given_x, atol=1e-2), "P(X, Y) != P(Y|X) * P(X)"
    assert np.allclose(joint_hist, joint_reconstructed_x_given_y, atol=1e-2), "P(X, Y) != P(X|Y) * P(Y)"

    print("Test passed: P(X,Y) == P(Y|X) * P(X) and P(X,Y) == P(X|Y) * P(Y)")

# Run the test
test_scm_sampling()