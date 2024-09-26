import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

def gradient_boost(x, y, number, lr, count=1, regs=[], foo=None):
    if number == 0:
        return
    else:
        if count > 1:
            y = y - regs[-1].predict(x)
        else:
            foo = y
        
        tree_reg = DecisionTreeRegressor(max_depth=5, random_state=42)
        tree_reg.fit(x, y)
        regs.append(tree_reg)
        
        x1 = np.linspace(-0.5, 0.5, 500)
        y_pred = sum(lr * regressor.predict(x1.reshape(-1, 1)) for regressor in regs)
        print(f"Iteration: {count}")
        
        plt.figure()
        plt.plot(x1, y_pred, linewidth=2)
        plt.plot(x[:, 0], foo, "r.")
        plt.title(f"Gradient Boosting Iteration {count}")
        plt.show()
        
        gradient_boost(x, y, number-1, lr, count+1, regs, foo=foo)

if __name__ == "__main__":
    np.random.seed(42)
    x = np.random.rand(100, 1) - 0.5
    y = 3 * x[:, 0]**2 + 0.05 * np.random.randn(100)
    gradient_boost(x, y, 5, lr=1)
