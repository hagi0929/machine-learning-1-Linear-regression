import numpy as np
load_data = np.loadtxt('./data-01-test-score.scv', delimiter=',', dtyple=np.float32)

x_raw = load_data[:, 0:-1]
r_raw = load_data[:, [-1]]
xp = np.array(x_raw).reshape(5, 3)
rp = np.array(r_raw).reshape(5, 1)

w = np.random.rand(3, 1)
b = np.random.rand(1)
print(f'w = {w} b = {b}')


def numerical_derivative(f, num):
    lgz = 1e-4
    der_ans = np.zeros_like(num)

    it = np.nditer(num, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        coor = it.multi_index
        tmp_val = num[coor]
        num[coor] = float(tmp_val) + lgz
        fx1 = f(num)

        num[coor] = tmp_val - lgz
        fx2 = f(num)
        der_ans[coor] = (fx1 - fx2) / (lgz * 2)

        num[coor] = tmp_val
        it.iternext()

    return der_ans


def error_val(jjj, t):
    y = np.dot(jjj, w) + b

    return np.sum((t - y) ** 2) / (len(jjj))


def loss_function(jjj, t):
    y = np.dot(jjj, w) + b

    return np.sum((t - y) ** 2) / len(jjj)


def predict(x):
    p = np.dot(x, w) + b

    return p


learning_rate: float = 1e-2
f = lambda x: loss_function(xp, rp)
print(f"initial error value: {error_val(xp, rp)} w= {w} b= {b}")

for step in range(8001):

    w -= learning_rate * numerical_derivative(f, w)
    b -= learning_rate * numerical_derivative(f, b)

    if step % 1000 == 0:
        print(f"step= {step} error value: {error_val(xp, rp)} w= {w} b= {b}")
test_data = np.array([100, 98, 81])

print(predict(test_data))