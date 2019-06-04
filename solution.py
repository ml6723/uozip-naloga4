# Preimenujte datoteko v solution.py, da bodo testi delovali

import numpy
from scipy.optimize import fmin_l_bfgs_b
from matplotlib import pylab

def draw_decision(lambda_, X, y, classifier, at1, at2, grid=50,):

    points = numpy.take(X, [at1, at2], axis=1)
    maxx, maxy = numpy.max(points, axis=0)
    minx, miny = numpy.min(points, axis=0)
    difx = maxx - minx
    dify = maxy - miny
    maxx += 0.02*difx
    minx -= 0.02*difx
    maxy += 0.02*dify
    miny -= 0.02*dify

    for c,(x,y) in zip(y,points):
        pylab.text(x,y,str(c), ha="center", va="center")
        pylab.scatter([x],[y],c=["b","r"][c!=0], s=200)

    num = grid
    prob = numpy.zeros([num, num])
    for xi,x in enumerate(numpy.linspace(minx, maxx, num=num)):
        for yi,y in enumerate(numpy.linspace(miny, maxy, num=num)):
            #probability of the closest example
            diff = points - numpy.array([x,y])
            dists = (diff[:,0]**2 + diff[:,1]**2)**0.5 #euclidean
            ind = numpy.argsort(dists)
            prob[yi,xi] = classifier(X[ind[0]])[1]

    pylab.imshow(prob, extent=(minx,maxx,maxy,miny))

    pylab.xlim(minx, maxx)
    pylab.ylim(miny, maxy)
    pylab.xlabel(at1)
    pylab.ylabel(at2)

    pylab.title(lambda_)

    pylab.show()


def load(name):
    """
    Odpri datoteko. Vrni matriko primerov (stolpci so znacilke)
    in vektor razredov.
    """
    data = numpy.loadtxt(name)
    X, y = data[:,:-1], data[:,-1].astype(numpy.int)
    return X,y

def load_csv(name):
    data = numpy.genfromtxt(name, dtype=str, delimiter=',')

    X = data[3:, :-2].astype(numpy.float)

    y = [float(1) for i in range(0,len(data[3:, -2]))]

    for i in range(0,len(y)):
        if data[3:, -2][i] == 'Boston':
            y[i] = float(0)

    image_names = data[3:, -1]

    return X, y, image_names

def h(x, theta):
    """
    Napovej verjetnost za razred 1 glede na podan primer (vektor vrednosti
    znacilk) in vektor napovednih koeficientov theta.
    """

    probability = 1 / (1 + numpy.exp(-(theta.T.dot(x))))

    if probability == 1:
        probability = 0.99999
    elif probability == 0:
        probability = 0.00001

    return probability

def cost(theta, X, y, lambda_):
    """
    Vrednost cenilne funkcije.
    """
    c = sum((y[i] * numpy.log(h(X[i], theta)) + (1-y[i]) * numpy.log((1-h(X[i], theta)))) for i in range(0,len(X)))
    c = c * -1/len(X)
    return c + (lambda_ / len(X)) * theta.dot(theta)

def grad(theta, X, y, lambda_):
    """
    Odvod cenilne funkcije. Vrne numpyev vektor v velikosti vektorja theta.
    """

    diffs = numpy.array([y[i] - h(X[i], theta) for i in range(0,len(X))])
    gradient = numpy.array([diffs.dot(X.T[i]) * -1/len(X) + lambda_ / len(X) * 2 * theta[i] for i in range(0,len(X[0]))])

    return gradient

class LogRegClassifier(object):

    def __init__(self, th):
        self.th = th

    def __call__(self, x):
        """
        Napovej razred za vektor vrednosti znacilk. Vrni
        seznam [ verjetnost_razreda_0, verjetnost_razreda_1 ].
        """
        x = numpy.hstack(([1.], x))
        p1 = h(x, self.th) #verjetno razreda 1
        return [1-p1, p1]

class LogRegLearner(object):

    def __init__(self, lambda_=0.0):
        self.lambda_ = lambda_

    def __call__(self, X, y):
        """
        Zgradi napovedni model za ucne podatke X z razredi y.
        """
        X = numpy.hstack((numpy.ones((len(X),1)), X))

        #optimizacija
        theta = fmin_l_bfgs_b(cost,
            x0=numpy.zeros(X.shape[1]),
            args=(X, y, self.lambda_),
            fprime=grad)[0]

        return LogRegClassifier(theta)


def test_learning(learner, X, y):
    """ vrne napovedi za iste primere, kot so bili uporabljeni pri učenju.
    To je napačen način ocenjevanja uspešnosti!

    Primer klica:
        res = test_learning(LogRegLearner(lambda_=0.0), X, y)
    """
    c = learner(X,y)
    results = [c(x) for x in X]
    return results


def test_cv(learner, X, y, seed=1, k=5):
    indexes = [i for i in range(0,len(X))]
    numpy.random.seed(seed)
    numpy.random.shuffle(indexes)

    predicted = {i: 0 for i in indexes}

    to_split = [0 for i in range(0,k)]


    for i in range(0,len(X)):
        to_split[i%k] += 1

    # TO - DO
    # kako se razdelijo v test_ind, če niso deljivi !!!

    for i in range(0,k):
        test_ind = indexes[sum(to_split[:i]): sum(to_split[:i]) + to_split[i]]
        test_X = numpy.array([X[j] for j in test_ind])
        train_X = numpy.array([X[j] for j in range(0,len(X)) if j not in test_ind])
        train_y = numpy.array([y[j] for j in range(0,len(X)) if j not in test_ind])

        classifier = learner(train_X, train_y)

        probabilities = []

        for j in range(0,to_split[i]):
            probabilities.append(classifier(test_X[j]))

        for j in range(0,to_split[i]):
            predicted[test_ind[j]] = probabilities[j]

    result = [0 for i in range(0,len(X))]

    for i in predicted.keys():
        result[i] = predicted[i]

    return numpy.array(result)


def CA(real, predictions):
    good_predictions = 0

    for i in range(0,len(real)):
        if (predictions[i][1] >= predictions[i][0]) & (real[i] == 1):
            good_predictions += 1
        elif (predictions[i][0] > predictions[i][1]) & (real[i] == 0):
            good_predictions += 1

    return good_predictions / len(real)


def AUC(real, predictions):
    index_class_one = numpy.array([i for i in range(0, len(real)) if real[i] == 1])
    index_class_zero = numpy.array([i for i in range(0, len(real)) if real[i] == 0])

    combinations = numpy.transpose([numpy.tile(index_class_one, len(index_class_zero)), numpy.repeat(index_class_zero, len(index_class_one))])

    auc = 0

    for c in combinations:
        index_one, index_zero = c[0], c[1]

        if predictions[index_one][1] > predictions[index_zero][1]:
            auc += 1
        elif predictions[index_one][1] == predictions[index_zero][1]:
            auc += 0.5

    return auc/len(combinations)


if __name__ == "__main__":
    # Primer uporabe

    X,y = load('reg.data')

    learner = LogRegLearner(lambda_=0.0)
    classifier = learner(X,y) # dobimo model

    napoved = classifier(X[0])  # napoved za prvi primer
    print(napoved)


