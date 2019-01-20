from sklearn.linear_model import LinearRegression

def train(x, y):
    model = LinearRegression().fit(x,y)
    return model

if __name__ == '__main__':
    model = train(x,y)
    x_new = 23.0
    y_new = model.predict(x_new)
    print(y_new)
