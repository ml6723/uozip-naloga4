import solution as s


# # Part 2:
# # za ta del naloge je treba program zagnati iz terminala
#
# X, y = s.load('reg.data')
#
# lambdas = [10, 0.03, 0.]
#
# for i in lambdas:
#     learner = s.LogRegLearner(lambda_=i)
#     classifier = learner(X, y)
#
#
#     s.draw_decision(i, X, y, classifier, 0, 1)


# Part 3 test_cv:

X, y = s.load('reg.data')

lambdas = [10, 1, 0.5, 0.1, 0.075, 0.05, 0.03, 0.01, 0.001, 0.0001, 0.]

lambdas_ca = {l: 0 for l in lambdas}

for i in range(1,21):
    for l in lambdas:
        learner = s.LogRegLearner(lambda_=l)
        res = s.test_cv(learner, X, y, seed=i)

        ca = s.CA(y, res)

        lambdas_ca[l] += ca

for l in lambdas_ca.keys():
    print('Lambda:', l)
    print('Tocnost', lambdas_ca[l]/20)


# # Part 3 test_learning:
#
# X, y = s.load('reg.data')
#
# lambdas = [10, 1, 0.5, 0.1, 0.05, 0.03, 0.01, 0.001, 0.0001, 0.]
#
# lambdas_ca = {l: 0 for l in lambdas}
#
#
# for l in lambdas:
#     learner = s.LogRegLearner(lambda_=l)
#     res = s.test_learning(learner, X, y)
#
#     ca = s.CA(y, res)
#
#     lambdas_ca[l] += ca
#
# for l in lambdas_ca.keys():
#     print('Lambda:', l)
#     print('Tocnost', lambdas_ca[l])


# # Part 4:
#
# X, y, image_names = s.load_csv('dogs.csv')
#
# # learner = s.LogRegLearner(lambda_=1)
# # res = s.test_cv(learner, X, y)
# #
# # ca = s.CA(y, res)
# #
# # print(ca)
#
# lambdas = [100, 10, 1, 0.5, 0.1, 0.05, 0.03, 0.01, 0.001, 0.0001, 0.]
#
# lambdas_ca = {l: 0 for l in lambdas}
#
# for i in range(1,11):
#     print(i)
#     for l in lambdas:
#         learner = s.LogRegLearner(lambda_=l)
#         res = s.test_cv(learner, X, y, seed=i+10)
#
#         ca = s.CA(y, res)
#
#         lambdas_ca[l] += ca
#
# for l in lambdas_ca.keys():
#     print('Lambda:', l)
#     print('Tocnost', lambdas_ca[l]/10)


# # Part bonus:
#
# # X, y = s.load('reg.data')
#
# X, y, image_names = s.load_csv('dogs.csv')
#
# learner = s.LogRegLearner(lambda_=1)
# res = s.test_cv(learner, X, y)
#
# auc = s.AUC(y, res)
#
# print(auc)