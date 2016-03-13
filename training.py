import time

from data_iterator import load_dataset, iterate_minibatches


def train(f_train_l, f_train_r, f_train_b,
          f_valid_l, f_valid_r, f_valid_b,
          xl, xr, y, learning_rate, num_epochs=20):

    # load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err_l = 0
        train_err_r = 0
        train_err_b = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
            inputs, targets = batch
            train_err_l += f_train_l(inputs, targets)
            train_err_r += f_train_r(inputs, targets)
            train_err_b += f_train_b(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
            inputs, targets = batch
            err, acc = f_valid_l(inputs, targets)
            err, acc = f_valid_r(inputs, targets)
            err, acc = f_valid_b(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss left:\t\t{:.6f}"
              .format(train_err_l / train_batches))
        print("  training loss right:\t\t{:.6f}"
              .format(train_err_r / train_batches))
        print("  training loss both:\t\t{:.6f}"
              .format(train_err_b / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
        inputs, targets = batch
        err, acc = f_valid_l(inputs, targets)
        err, acc = f_valid_r(inputs, targets)
        err, acc = f_valid_b(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))
