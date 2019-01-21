# We're looking to find the local minimum of the function:
#
#   f(x)  = x^4  - 3x^3 + 2
#   f'(x) = 4x^3 - 9x^2
#
# Given a starting value x, the algorithm will follow the
# gradient at x until a local minimum value is found.
#
# We use gradient descent rather than direct calculation because
# it is computationally cheaper for large amount of data and
# because it is a more general method.

def gradient_descent(function, cur_x, epoch, precision):
    gamma              = 0.01  # step size multiplier (learning rate)
    previous_step_size = 1
    iters              = 0     # iteration counter
    while previous_step_size > precision and iters < epoch:
        prev_x = cur_x
        cur_x -= gamma * function(prev_x)
        previous_step_size = abs(cur_x - prev_x)
        iters += 1
    return cur_x

if __name__ == '__main__':

    # f'(x) = 4x^3 - 9x^2
    df = lambda x: 4 * x**3 - 9 * x**2
    
    # The algorithm starts at x = 6
    # Iterate 10000 times
    # With a precision of 0.00001
    x = gradient_descent(df, 6, 10000, 0.00001)
    
    print("The local minimum occurs at", x)
    # The output for the above will be: ('The local minimum occurs at', 2.2499646074278457)
