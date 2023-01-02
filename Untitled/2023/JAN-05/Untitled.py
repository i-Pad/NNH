from mnist import MNIST

mnist = MNIST('../data/MNIST')

X_train, y_train = mnist.load_training()
X_test, y_test = mnist.load_testing()

print(len(X_train))
print(type(X_train[0]))
