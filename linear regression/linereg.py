def update_w_and_b(spendings, sales, w, b, alpha):
    dl_dw = 0.0
    dl_db = 0.0
    N = len(spendings)
    for i in range(N):
        output = predict(spendings[i], w, b)
        error  = sales[i] - output
        dl_dw  += spendings[i] * error
        dl_db  += error
    # update w and b
    w += (2/float(N)) * dl_dw * alpha
    b += (2/float(N)) * dl_db * alpha
    return w, b

def avg_loss(spendings, sales, w, b):
    N = len(spendings)
    total_error = 0.0
    for i in range(N):
        output = predict(spendings[i], w, b)
        error  = sales[i] - output
        total_error += error**2
    return total_error / float(N)

def train(spendings, sales, w, b, alpha, epochs):
    for e in range(epochs):
        w, b = update_w_and_b(spendings, sales, w, b, alpha)
        # log the progress
        if e % 400 == 0:
            print("epoch:", e, "loss: ", avg_loss(spendings, sales, w, b))
    return w, b

def predict(x, w, b):
    return w * x + b

if __name__ == '__main__':
    w, b = train(x, y, 0.0, 0.0, 0.001, 15000)
    x_new = 23.0
    y_new = predict(x_new, w, b)
    print(y_new)
