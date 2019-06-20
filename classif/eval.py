
# import pandas as pd
import numpy as np
# gt = pd.read_csv('solution.csv')
# predictions = pd.read_csv('sample_submission.csv')
#
import pickle
import os

def avg_precision(actual=None, predicted=None):
    """Computes average label precision. function my Manoj Plakal, but here is done with strings"""
    for (i, p) in enumerate(predicted):
        if actual == p:
            return 1.0 / (i + 1.0)
    return 0.0


def get_accuracy(actual=None, predicted=None):
    """Computes accuracy, done with strings"""
    if predicted == actual:
        return 1.0
    else:
        return 0.0


class Evaluator (object):

    def __init__(self, gt=None, predictions=None, list_labels=None, params_ctrl=None, params_files=None):
        self.gt = gt
        self.predictions = predictions
        self.list_labels = list_labels
        self.path_results = params_files['results']

        if not os.path.isfile(params_files['results']):
            # create empty dict for results and dump pickle
            results_asdict = {cat_name: [] for cat_name in list_labels}
            results_asdict['overall'] = []
            pickle.dump(results_asdict, open(self.path_results, "wb"))

    def evaluate_acc(self):
        """
        input two dataframes to compare
        :param gt:
        :param predictions:
        :return:
        """
        print('\n=====Evaluating ACCURACY - MICRO =============================')
        acc = {}
        for index, row in self.predictions.iterrows():
            pred_per_file = row['label']
            temp = self.gt.loc[self.gt['fname'] == row['fname']]
            for idx_gt, row_gt in temp.iterrows():
                acc[row_gt['fname']] = get_accuracy(actual=row_gt['label'], predicted=pred_per_file)

        sum_acc = 0
        for f_name, score in acc.items():
            sum_acc += score
        self.mean_acc = (sum_acc / len(acc))*100
        print('Number of files evaluated: %d' % len(acc))
        print('Mean Accuracy for files evaluated: %6.3f' % self.mean_acc)
        results_asdict = pickle.load(open(self.path_results, 'rb'))
        results_asdict['overall'].append(self.mean_acc)
        pickle.dump(results_asdict, open(self.path_results, "wb"))


    def evaluate_acc_classwise(self):

        print('\n=====Evaluating ACCURACY - PER CLASS ======================================================')
        # init with nested dicts
        scores = {key: {'nb_files': 0, 'acc_cum': 0} for key in self.list_labels}

        for idx_gt, row_gt in self.gt.iterrows():
            # print(predictions.loc[predictions['fname'] == row_gt['fname']])
            predicted_match = self.predictions.loc[self.predictions['fname'] == row_gt['fname']]
            for idx_pred, row_pred in predicted_match.iterrows():
                pred_per_file = row_pred['label']
                scores[row_gt['label']]['nb_files'] += 1
                # computing ACCURACY and saving it in the due class
                scores[row_gt['label']]['acc_cum'] += get_accuracy(actual=row_gt['label'], predicted=pred_per_file)

        total = 0
        perclass_acc = []
        results_asdict = pickle.load(open(self.path_results, 'rb'))
        for label, v in scores.items():
            mean_acc = (v['acc_cum'] / v['nb_files'])*100
            print('%-21s | number of files in total: %-4d | Accuracy: %6.3f' % (label, v['nb_files'], mean_acc))
            perclass_acc.append(mean_acc)
            total += v['nb_files']
            results_asdict[label].append(mean_acc)
        print('Total number of files: %d' % total)
        pickle.dump(results_asdict, open(self.path_results, "wb"))

        print('\n=====Printing sorted classes for ACCURACY - PER CLASS ========================================')
        perclass_acc_np = np.array(perclass_acc)
        idx_sort = np.argsort(-perclass_acc_np)
        for i in range(len(self.list_labels)):
            print('%-21s | number of files in total: %-4d | Accuracy: %6.3f' %
                  (self.list_labels[idx_sort[i]], scores[self.list_labels[idx_sort[i]]]['nb_files'],
                   perclass_acc[idx_sort[i]]))

    def print_summary_eval(self):
        """
        just print the metrics for the PRIVATE leaderboard together, so that it is easier to inspect
        :return:
        """
        print('\n=====================================================================================================')
        print('=====================================================================================================')
        print('SUMMARY of evaluation:')
        print('Mean Accuracy for files evaluated: %5.2f' % self.mean_acc)
        print('\n=====================================================================================================')

        if self.count_trial >= 6:
            # print summary of results across all the trials
            results_asdict = pickle.load(open(self.path_results, 'rb'))
            print('\n======================================================================================')
            print('========================================================================================')
            print('SUMMARY of evaluation across ALL THE TRIALS processed so far for the {0} SUBSET of train data:'.format(self.train_data))
            print('\n======================================================================================')
            for label, accs in results_asdict.items():
                accs = np.array(accs)
                mean_accs = np.mean(accs)
                std_accs = np.std(accs)
                if label == 'overall':
                    print('\nMean (micro):')
                print('%-22s | Mean Accuracy / stdev: %6.2f %6.2f' % (label, mean_accs, std_accs))




