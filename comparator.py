from gaussian import Util


class Comparator:

    def __init__(self, number_gaussian):
        self.ng = number_gaussian

    def sumary(self, data_test, target_test, models):
        accepts, impostors = self.compare(data_test, target_test, models)

        taxa_accepts = sum(accepts) / len(data_test)
        taxa_impostors = sum(impostors) / len(data_test)
        taxa_errors = (len(data_test) - sum(accepts)) / len(data_test)

        print("Taxa de acertos: %f" % taxa_accepts)
        print("Taxa de errors: %f" % taxa_errors)
        print("Taxa de impostors: %f" % taxa_impostors)

        return taxa_accepts, taxa_errors, taxa_impostors

    def compare(self, data_test, target_test, models):
        accepts = []
        impostors = []
        for ix in range(0, len(data_test)):
            data = data_test[ix]
            target = target_test[ix]
            best_class = self.find_best_class(data, models)

            client, impostor = models[best_class]
            isclient = self.compare_with_impostor(data, client, impostor)

            impostors.append(not isclient)
            accepts.append(isclient and (best_class == target))

        return accepts, impostors

    def find_best_class(self, data, models):
        util = Util(self.ng)
        best_p = 0
        best_class = -1
        for c in range(0, len(models)):
            client, impostor = models[c]
            p = util.sum_weighted_gauss(data, client)
            if p > best_p:
                best_p = p
                best_class = c
        return best_class

    def compare_with_impostor(self, data, client, impostor):
        util = Util(self.ng)
        prob_client = util.sum_weighted_gauss(data, client)
        prob_impostor = util.sum_weighted_gauss(data, impostor)

        return prob_client > prob_impostor