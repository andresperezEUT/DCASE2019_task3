
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

    def evaluate_map3(self):
        """
        Computes overall MAP@3 based on two dataframes with ground truth and predictions
        :param gt:
        :param predictions:
        :return:
        """
        print('\n=====Evaluating MAP@3 - MICRO =====================================================================')
        ap_pri = {}
        ap_pub = {}

        for index, row in self.predictions.iterrows():
            # print(index, row['fname'], row['label'])
            preds_per_file = row['label'].split(' ')

            temp = self.gt.loc[self.gt['fname'] == row['fname']]

            for idx_gt, row_gt in temp.iterrows():
                # print(idx_gt, row_gt['fname'], row_gt['label'], row_gt['Usage'])
                if row_gt['Usage'] == 'Private':
                    # computing MAP3 only over the private files
                    ap_pri[row_gt['fname']] = avg_precision(actual=row_gt['label'], predicted=preds_per_file)

                if row_gt['Usage'] == 'Public':
                    # computing MAP3 only over the private files
                    ap_pub[row_gt['fname']] = avg_precision(actual=row_gt['label'], predicted=preds_per_file)

        # could make a function of this
        sum_ap_pri = 0
        # all the test files have been evaluated.
        for f_name, score in ap_pri.items():
            sum_ap_pri += score

        self.MAP_3_pri = sum_ap_pri / len(ap_pri)
        print('Number of PRIVATE files evaluated: %d' % len(ap_pri))
        print('MAP_3 for PRIVATE files evaluated: %7.5f' % self.MAP_3_pri)

        # --
        sum_ap_pub = 0
        # all the test files have been evaluated.
        for f_name, score in ap_pub.items():
            sum_ap_pub += score

        if len(ap_pub) > 0:
            self.MAP_3_pub = sum_ap_pub / len(ap_pub)
            print('\nNumber of PUBLIC files evaluated: %d' % len(ap_pub))
            print('MAP_3 for PUBLIC files evaluated: %7.5f' % self.MAP_3_pub)

    def evaluate_map3_classwise(self):
        """
        input two dataframes to compare
        :param gt:
        :param predictions:
        :return:
        compute only on private (1299 files)
        doing it on the public one may be misleading (only 301 files, ie few files per class)
        """
        print('\n=====Evaluating MAP@3 - PER CLASS - PRIVATE ============================================================')
        scores_pri = {}
        # init with nested dicts
        for i in range(len(self.list_labels)):
            scores_pri[self.list_labels[i]] = {}
            scores_pri[self.list_labels[i]]['nb_files'] = 0
            scores_pri[self.list_labels[i]]['AP_3cum'] = 0

        for idx_gt, row_gt in self.gt.iterrows():
            if row_gt['Usage'] == 'Private':

                # print(predictions.loc[predictions['fname'] == row_gt['fname']])
                predicted_match = self.predictions.loc[self.predictions['fname'] == row_gt['fname']]
                for idx_pred, row_pred in predicted_match.iterrows():
                    preds_per_file = row_pred['label'].split(' ')

                    scores_pri[row_gt['label']]['nb_files'] += 1

                    # computing MAP3 and saving it in the due class
                    scores_pri[row_gt['label']]['AP_3cum'] += avg_precision(actual=row_gt['label'], predicted=preds_per_file)

        total = 0
        perclass_mAP = []
        for label, data_map in scores_pri.items():
            if data_map['nb_files'] > 0:
                print('%-21s | number of private files in total: %-3d | MAP@3: %7.5f' %
                      (label, data_map['nb_files'], data_map['AP_3cum']/data_map['nb_files']))

                perclass_mAP.append(data_map['AP_3cum'] / data_map['nb_files'])

            else:
                print('%-21s | number of private files in total: %-3d | MAP@3: %7.5f' %
                      (label, data_map['nb_files'], 0.0))
            total += data_map['nb_files']

        print('Total number of files: %d \n' % total)

        print('\n=====Printing sorted classes for MAP_3 - PER CLASS - PRIVATE ========================================')
        perclass_mAP_np = np.array(perclass_mAP)
        # top_3 = list_labels[np.argsort(-te_preds, axis=1)[:, :3]]
        idx_sort = np.argsort(-perclass_mAP_np)

        for i in range(len(self.list_labels)):
            if scores_pri[self.list_labels[idx_sort[i]]]['nb_files'] > 0:
                print('%-21s | number of private files in total: %-3d | MAP@3: %7.5f' %
                      (self.list_labels[idx_sort[i]], scores_pri[self.list_labels[idx_sort[i]]]['nb_files'],
                       perclass_mAP[idx_sort[i]]))

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

    def evaluate_acc_shy(self):
        """
        input two dataframes to compare
        :param gt:
        :param predictions:
        :return:
        """
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
        return self.mean_acc

    def evaluate_acc_classwise(self):
        """
        input two dataframes to compare
        :param gt:
        :param predictions:
        :return:
        compute only on private (1299 files)
        doing it on the public one may be misleading (only 301 files, ie few files per class)
        """
        print('\n=====Evaluating ACCURACY - PER CLASS ======================================================')
        scores = {}
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




