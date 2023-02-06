import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score
from calculate_relative import *
from ER_rule import *
from sklearn import metrics
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import rpy2.robjects as robj
import scipy.stats as st

def roc_test_r(targets_1, scores_1, targets_2, scores_2, method='delong'):
    # method: “delong”, “bootstrap” or “venkatraman”
    importr('pROC')
    robj.globalenv['targets_1'] = targets_1 = robj.FloatVector(targets_1)
    robj.globalenv['scores_1'] = scores_1 = robj.FloatVector(scores_1)
    robj.globalenv['targets_2'] = targets_2 = robj.FloatVector(targets_2)
    robj.globalenv['scores_2'] = scores_2 = robj.FloatVector(scores_2)
    robj.r('roc_1 <- roc(targets_1, scores_1)')
    robj.r('roc_2 <- roc(targets_2, scores_2)')
    robj.r('result = roc.test(roc_1, roc_2, method="%s")' % method)
    p_value = robj.r('p_value = result$p.value')
    return np.array(p_value)[0]


def auc_ci(target, score):
    importr('pROC')
    robj.globalenv['target'] = target = robj.FloatVector(target)
    robj.globalenv['score'] = score = robj.FloatVector(score)
    robj.r('rocobj <- roc(target, score)')
    result = robj.r('result <- ci.auc(rocobj)')
    return result


def plot_cal_r(targets_1, scores_1, targets_2, scores_2):
    importr('rms')
    importr('grid')
    importr('lattice')
    importr('ggplot2')
    robj.globalenv['targets_1'] = targets_1 = robj.FloatVector(targets_1)
    robj.globalenv['scores_1'] = scores_1 = robj.FloatVector(scores_1)
    robj.globalenv['targets_2'] = targets_2 = robj.FloatVector(targets_2)
    robj.globalenv['scores_2'] = scores_2 = robj.FloatVector(scores_2)
    robj.r('f1 <- lrm(formula=targets_1~scores_1, data = data.frame(targets_1, scores_1), x=T, y=T)')
    robj.r('cal2 <- calibrate(f1, cmethod="hare", method="boot", B=1000, data=data.frame(targets_2, scores_2))')
    robj.r('plot(cal2,xlim=c(0,1.0),ylim=c(0,1.0)')

T4 = pd.read_excel('Tabel.xlsx', index_col=4)
train_i = T4[T4["Date of Surgery"] < '2018-03-01'].index.tolist()
test_i = T4[T4["Date of Suegery"] >= '2018-03-01'].index.tolist()

T4_train = T4[T4.index.isin(train_i)]
T4_test = T4[T4.index.isin(test_i)]

T4.loc[:, "pP"] = T4.loc[:, "pP"].map({"P0": 0, "P1": 1})
T4.loc[:, "cP"] = T4.loc[:, "cP"].map({"P0": 0, "P1": 1})

Y_train = T4[T4.index.isin(train_i)]['pP']
Y_test = T4[T4.index.isin(test_i)]['pP']

Y_CT_train = T4[T4.index.isin(train_i)]['CT_score']
Y_CT_test = T4[T4.index.isin(test_i)]['CT_score']

Y_PET_train = T4[T4.index.isin(train_i)]['PET_score']
Y_PET_test = T4[T4.index.isin(test_i)]['PET_score']

Y_Hu_train = T4[T4.index.isin(train_i)]['cP']
Y_Hu_test = T4[T4.index.isin(test_i)]['cP']

Y_Cl_train = T4[T4.index.isin(train_i)]['Clinical']
Y_Cl_test = T4[T4.index.isin(test_i)]['Clinical']

Y_CT_Rad_train = T4[T4.index.isin(train_i)]['CT_Rad']
Y_CT_Rad_test = T4[T4.index.isin(test_i)]['CT_Rad']

Y_PET_Rad_train = T4[T4.index.isin(train_i)]['PET_Rad']
Y_PET_Rad_test = T4[T4.index.isin(test_i)]['PET_Rad']

Y_CT_KSTM_train = T4[T4.index.isin(train_i)]['CT_KSTM']
Y_CT_KSTM_test = T4[T4.index.isin(test_i)]['CT_KSTM']

Y_PET_KSTM_train = T4[T4.index.isin(train_i)]['PET_KSTM']
Y_PET_KSTM_test = T4[T4.index.isin(test_i)]['PET_KSTM']



X_CT_train = pd.concat([Y_CT_Rad_train, Y_CT_KSTM_train], axis=1)
L = X_CT_train.values.tolist()
w = [0.82857, 0.8332]
t = [0.4666, 0.2942]
Y_CT_cb_train = np.zeros(len(L))
for i in range(len(L)):
    P = [[1 - L[i][0], L[i][0]],
         [1 - L[i][1], L[i][1]]
         ]
    R = calculating_reliability_new(P, t)
    P_fin = Analytic_ER_rule(R, w, P)
    Y_CT_cb_train[i] = P_fin[1]

X_CT_test = pd.concat([Y_CT_Rad_test, Y_CT_KSTM_test], axis=1)
L = X_CT_test.values.tolist()
w = [0.82857, 0.8332]
t = [0.4666, 0.2942]
Y_CT_cb_test = np.zeros(len(L))
for i in range(len(L)):
    P = [[1 - L[i][0], L[i][0]],
         [1 - L[i][1], L[i][1]]
         ]
    R = calculating_reliability_new(P, t)
    P_fin = Analytic_ER_rule(R, w, P)
    Y_CT_cb_test[i] = P_fin[1]

Y_CT_pred_train = pd.DataFrame(data=Y_CT_cb_train, columns=['CT_score'], index=train_i)
Y_CT_pred_test = pd.DataFrame(data=Y_CT_cb_test, columns=['CT_score'], index=test_i)
score = pd.concat([Y_CT_pred_train, Y_CT_pred_test])
# score.to_csv('CT final score.csv')

X_PET_train = pd.concat([Y_PET_Rad_train, Y_PET_KSTM_train], axis=1)
L = X_PET_train.values.tolist()
w = [0.87369, 0.7498]
t = [0.4120, 0.5316]
Y_PET_cb_train = np.zeros(len(L))
for i in range(len(L)):
    P = [[1 - L[i][0], L[i][0]],
         [1 - L[i][1], L[i][1]]
         ]
    R = calculating_reliability_new(P, t)
    P_fin = Analytic_ER_rule(R, w, P)
    Y_PET_cb_train[i] = P_fin[1]

X_PET_test = pd.concat([Y_PET_Rad_test, Y_PET_KSTM_test], axis=1)
L = X_PET_test.values.tolist()
w = [0.87369, 0.7498]
t = [0.4120, 0.5316]
Y_PET_cb_test = np.zeros(len(L))
for i in range(len(L)):
    P = [[1 - L[i][0], L[i][0]],
         [1 - L[i][1], L[i][1]]
         ]
    R = calculating_reliability_new(P, t)
    P_fin = Analytic_ER_rule(R, w, P)
    Y_PET_cb_test[i] = P_fin[1]

Y_PET_pred_train = pd.DataFrame(data=Y_PET_cb_train, columns=['PET_score'], index=train_i)
Y_PET_pred_test = pd.DataFrame(data=Y_PET_cb_test, columns=['PET_score'], index=test_i)
score = pd.concat([Y_PET_pred_train, Y_PET_pred_test])
# score.to_csv('PET final score.csv')

X_cb_train = pd.concat([Y_CT_train, Y_PET_train, Y_Cl_train, Y_Hu_train], axis=1)
L = X_cb_train.values.tolist()
w = [0.89686, 0.90371, 0.86261, 0.70407]
t = [0.3128, 0.4649, 0.4673, 0.5]
Y_cb_train = np.zeros(len(L))
for i in range(len(L)):
    P = [[1 - L[i][0], L[i][0]],
         [1 - L[i][1], L[i][1]],
         [1 - L[i][2], L[i][2]],
         [1 - L[i][3], L[i][3]]
         ]
    R = calculating_reliability_new(P, t)
    P_fin = Analytic_ER_rule(R, w, P)
    Y_cb_train[i] = P_fin[1]

X_cb_test = pd.concat([Y_CT_test, Y_PET_test, Y_Cl_test, Y_Hu_test], axis=1)
L = X_cb_test.values.tolist()
w = [0.89686, 0.90371, 0.86261, 0.70407]
t = [0.3128, 0.4649, 0.4673, 0.5]
Y_cb_test = np.zeros(len(L))
for i in range(len(L)):
    P = [[1 - L[i][0], L[i][0]],
         [1 - L[i][1], L[i][1]],
         [1 - L[i][2], L[i][2]],
         [1 - L[i][3], L[i][3]]
         ]
    R = calculating_reliability_new(P, t)
    P_fin = Analytic_ER_rule(R, w, P)
    Y_cb_test[i] = P_fin[1]

print('Fusion Model train vs test, Delong test: p=', roc_test_r(Y_train, Y_cb_train, Y_test, Y_cb_test))
print('Fusion Model vs CT train, Delong test: p=', roc_test_r(Y_train, Y_CT_train, Y_train, Y_cb_train))
print('Fusion Model vs CT test, Delong test: p=', roc_test_r(Y_test, Y_CT_test, Y_test, Y_cb_test))
print('Fusion Model vs PET train, Delong test: p=', roc_test_r(Y_train, Y_PET_train, Y_train, Y_cb_train))
print('Fusion Model vs PET test, Delong test: p=', roc_test_r(Y_test, Y_PET_test, Y_test, Y_cb_test))
print('Fusion Model vs Clinical Nomogram train, Delong test: p=', roc_test_r(Y_train, Y_Cl_train, Y_train, Y_cb_train))
print('Fusion Model vs Clinical Nomogram test, Delong test: p=', roc_test_r(Y_test, Y_Cl_test, Y_test, Y_cb_test))
print('Fusion Model vs Human train, Delong test: p=', roc_test_r(Y_train, Y_Hu_train, Y_train, Y_cb_train))
print('Fusion Model vs Human test, Delong test: p=', roc_test_r(Y_test, Y_Hu_test, Y_test, Y_cb_test))


def best_thres(fpr, tpr, threshold):
    max_Youden = tpr[0] - fpr[0]
    best_i = 0
    for i in range(len(threshold)):
        if tpr[i] - fpr[i] > max_Youden:
            max_Youden = tpr[i] - fpr[i]
            best_i = i
    return best_i

AUC_nomo_train = round(roc_auc_score(Y_train, Y_nomo_train) * 100, 3)
AUC_nomo_test = round(roc_auc_score(Y_test, Y_nomo_test) * 100, 3)
AUC_CT_train = round(roc_auc_score(Y_train, Y_CT_train) * 100, 3)
AUC_CT_test = round(roc_auc_score(Y_test, Y_CT_test) * 100, 3)
AUC_PET_train = round(roc_auc_score(Y_train, Y_PET_train) * 100, 3)
AUC_PET_test = round(roc_auc_score(Y_test, Y_PET_test) * 100, 3)
AUC_Hu_train = round(roc_auc_score(Y_train, Y_Hu_train) * 100, 3)
AUC_Hu_test = round(roc_auc_score(Y_test, Y_Hu_test) * 100, 3)
AUC_Cl_train = round(roc_auc_score(Y_train, Y_Cl_train) * 100, 3)
AUC_Cl_test = round(roc_auc_score(Y_test, Y_Cl_test) * 100, 3)
AUC_cb_train = round(roc_auc_score(Y_train, Y_cb_train) * 100, 3)
AUC_cb_test = round(roc_auc_score(Y_test, Y_cb_test) * 100, 3)

auc_Hu_train = auc_ci(Y_train, Y_Hu_train)
print('Human train', auc_Hu_train)
auc_Hu_test = auc_ci(Y_test, Y_Hu_test)
print('Human test', auc_Hu_test)
auc_Cl_train = auc_ci(Y_train, Y_Cl_train)
print('Clinical train', auc_Cl_train)
auc_Cl_test = auc_ci(Y_test, Y_Cl_test)
print('Clinical test', auc_Cl_test)
auc_nomo_train = auc_ci(Y_train, Y_nomo_train)
print('nomo train', auc_nomo_train)
auc_nomo_test = auc_ci(Y_test, Y_nomo_test)
print('nomo test', auc_nomo_test)
auc_CT_train = auc_ci(Y_train, Y_CT_train)
print('CT train', auc_CT_train)
auc_CT_test = auc_ci(Y_test, Y_CT_test)
print('CT test', auc_CT_test)
auc_PET_train = auc_ci(Y_train, Y_PET_train)
print('PET train', auc_PET_train)
auc_PET_test = auc_ci(Y_test, Y_PET_test)
print('PET test', auc_PET_test)
auc_cb_train = auc_ci(Y_train, Y_cb_train)
print('MMF train', auc_cb_train)
auc_cb_test = auc_ci(Y_test, Y_cb_test)
print('MMF test', auc_cb_test)

AUC_CT_Rad_train = round(roc_auc_score(Y_train, Y_CT_Rad_train) * 100, 3)
AUC_CT_Rad_test = round(roc_auc_score(Y_test, Y_CT_Rad_test) * 100, 3)
AUC_CT_KSTM_train = round(roc_auc_score(Y_train, Y_CT_KSTM_train) * 100, 3)
AUC_CT_KSTM_test = round(roc_auc_score(Y_test, Y_CT_KSTM_test) * 100, 3)
AUC_PET_Rad_train = round(roc_auc_score(Y_train, Y_PET_Rad_train) * 100, 3)
AUC_PET_Rad_test = round(roc_auc_score(Y_test, Y_PET_Rad_test) * 100, 3)
AUC_PET_KSTM_train = round(roc_auc_score(Y_train, Y_PET_KSTM_train) * 100, 3)
AUC_PET_KSTM_test = round(roc_auc_score(Y_test, Y_PET_KSTM_test) * 100, 3)


plt.figure(figsize=(6, 6), dpi=300)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
fpr, tpr, threshold = roc_curve(Y_train, Y_Hu_train)
Hu_i = best_thres(fpr, tpr, threshold)
Hu_thres = threshold[Hu_i]
plt.plot(fpr, tpr, color="green", alpha=0.5, lw=2,
         label='Experts\' Diagnoses (AUC = %0.2f %%)' % AUC_Hu_train)
plt.plot(fpr[Hu_i], tpr[Hu_i], "go")

fpr, tpr, threshold = roc_curve(Y_train, Y_Cl_train)
Cl_i = best_thres(fpr, tpr, threshold)
Cl_thres = threshold[Cl_i]
plt.plot(fpr, tpr, color="blue", alpha=0.5, lw=2,
         label='Clinical Nomogram (AUC = %0.2f %%)' % AUC_Cl_train)
plt.plot(fpr[Cl_i], tpr[Cl_i], "bo")

fpr, tpr, threshold = roc_curve(Y_train, Y_CT_train)
CT_i = best_thres(fpr, tpr, threshold)
CT_thres = threshold[CT_i]
plt.plot(fpr, tpr, color="darkorange", alpha=0.5, lw=2,
         label='CT Signature (AUC = %0.2f %%)' % AUC_CT_train)
plt.plot(fpr[CT_i], tpr[CT_i], "o", color='darkorange')

fpr, tpr, threshold = roc_curve(Y_train, Y_PET_train)
PET_i = best_thres(fpr, tpr, threshold)
PET_thres = threshold[PET_i]
plt.plot(fpr, tpr, color="red", alpha=0.5, lw=2,
         label='PET Signature (AUC = %0.2f %%)' % AUC_PET_train)
plt.plot(fpr[PET_i], tpr[PET_i], "ro")

fpr, tpr, threshold = roc_curve(Y_train, Y_cb_train)
cb_i = best_thres(fpr, tpr, threshold)
cb_thres = threshold[cb_i]
plt.plot(fpr, tpr, color="black", alpha=0.5, lw=2,
         label='MMF Model (AUC = %0.2f %%)' % AUC_cb_train)
plt.plot(fpr[cb_i], tpr[cb_i], "o", color='black')
plt.title('Training ROC Curve')
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
plt.legend()
plt.show()

plt.figure(figsize=(6, 6), dpi=300)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
fpr, tpr, threshold = roc_curve(Y_train, Y_CT_Rad_train)
CT_Rad_i = best_thres(fpr, tpr, threshold)
CT_Rad_thres = threshold[CT_Rad_i]
plt.plot(fpr, tpr, color="green", alpha=0.5, lw=2,
         label='CT Radiomic Model (AUC = %0.2f %%)' % AUC_CT_Rad_train)
plt.plot(fpr[CT_Rad_i], tpr[CT_Rad_i], "o", color='green')
fpr, tpr, threshold = roc_curve(Y_train, Y_CT_KSTM_train)
CT_KSTM_i = best_thres(fpr, tpr, threshold)
CT_KSTM_thres = threshold[CT_KSTM_i]
plt.plot(fpr, tpr, color="blue", alpha=0.5, lw=2,
         label='CT KSTM Model (AUC = %0.2f %%)' % AUC_CT_KSTM_train)
plt.plot(fpr[CT_KSTM_i], tpr[CT_KSTM_i], "o", color='blue')
fpr, tpr, threshold = roc_curve(Y_train, Y_CT_train)
CT_i = best_thres(fpr, tpr, threshold)
CT_thres = threshold[CT_i]
plt.plot(fpr, tpr, color='darkorange', alpha=0.5, lw=2,
         label='CT Signature (AUC = %0.2f %%)' % AUC_CT_train)
plt.plot(fpr[CT_i], tpr[CT_i], "o", color='darkorange')
plt.title('Training ROC Curve')
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
plt.legend()
plt.show()

plt.figure(figsize=(6, 6), dpi=300)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
fpr, tpr, threshold = roc_curve(Y_train, Y_PET_Rad_train)
PET_Rad_i = best_thres(fpr, tpr, threshold)
PET_Rad_thres = threshold[PET_Rad_i]
plt.plot(fpr, tpr, color="green", alpha=0.5, lw=2,
         label='PET Radiomic Model (AUC = %0.2f %%)' % AUC_PET_Rad_train)
plt.plot(fpr[PET_Rad_i], tpr[PET_Rad_i], "o", color='green')
fpr, tpr, threshold = roc_curve(Y_train, Y_PET_KSTM_train)
PET_KSTM_i = best_thres(fpr, tpr, threshold)
PET_KSTM_thres = threshold[PET_KSTM_i]
plt.plot(fpr, tpr, color="blue", alpha=0.5, lw=2,
         label='PET KSTM Model (AUC = %0.2f %%)' % AUC_PET_KSTM_train)
plt.plot(fpr[PET_KSTM_i], tpr[PET_KSTM_i], "o", color='blue')
fpr, tpr, threshold = roc_curve(Y_train, Y_PET_train)
PET_i = best_thres(fpr, tpr, threshold)
PET_thres = threshold[PET_i]
plt.plot(fpr, tpr, color="red", alpha=0.5, lw=2,
         label='PET Signature (AUC = %0.2f %%)' % AUC_PET_train)
plt.plot(fpr[PET_i], tpr[PET_i], "o", color='red')
plt.title('Training ROC Curve')
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
plt.legend()
plt.show()

def thres_index(thres, threshold):
    thres_i = 0
    for i in range(len(threshold)):
        if thres < threshold[i]:
            continue
        else:
            thres_i = i
            break
    return thres_i


def sensi_speci(thres, pred_prob, result):
    Y_pred = pred_prob.copy()
    Y_pred[Y_pred >= thres] = 1
    Y_pred[Y_pred < thres] = 0
    matrix = metrics.confusion_matrix(result, Y_pred)
    sensitivity = matrix[1][1] / (matrix[1][1] + matrix[1][0])
    specificity = matrix[0][0] / (matrix[0][0] + matrix[0][1])
    return sensitivity, specificity

plt.figure(figsize=(6, 6), dpi=300)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
fpr, tpr, threshold = roc_curve(Y_test, Y_Hu_test)
plt.plot(fpr, tpr, color="green", alpha=0.5, lw=2,
         label='Experts\' Diagnoses (AUC = %0.2f %%)' % AUC_Hu_test)
Hu_i = thres_index(Hu_thres, threshold)
sensitivity, specificity = sensi_speci(Hu_thres, Y_Hu_test, Y_test)
plt.plot(1 - specificity, sensitivity, "go")

fpr, tpr, threshold = roc_curve(Y_test, Y_Cl_test)
plt.plot(fpr, tpr, color="blue", alpha=0.5, lw=2,
         label='Clinical Nomogram (AUC = %0.2f %%)' % AUC_Cl_test)
Cl_i = thres_index(Cl_thres, threshold)
sensitivity, specificity = sensi_speci(Cl_thres, Y_Cl_test, Y_test)
plt.plot(1 - specificity, sensitivity, "bo")

fpr, tpr, threshold = roc_curve(Y_test, Y_CT_test)
plt.plot(fpr, tpr, color="darkorange", alpha=0.5, lw=2,
         label='CT Signature (AUC = %0.2f %%)' % AUC_CT_test)
CT_i = thres_index(CT_thres, threshold)
sensitivity, specificity = sensi_speci(CT_thres, Y_CT_test, Y_test)
plt.plot(1 - specificity, sensitivity, "o", color='darkorange')

fpr, tpr, threshold = roc_curve(Y_test, Y_PET_test)
plt.plot(fpr, tpr, color="red", alpha=0.5, lw=2,
         label='PET Signature (AUC = %0.2f %%)' % AUC_PET_test)
PET_i = thres_index(PET_thres, threshold)
sensitivity, specificity = sensi_speci(PET_thres, Y_PET_test, Y_test)
plt.plot(1 - specificity, sensitivity, "ro")

fpr, tpr, threshold = roc_curve(Y_test, Y_cb_test)
plt.plot(fpr, tpr, color='black', alpha=0.5, lw=2,
         label='MMF Model (AUC = %0.2f %%)' % AUC_cb_test)
cb_i = thres_index(cb_thres, threshold)
sensitivity, specificity = sensi_speci(cb_thres, Y_cb_test, Y_test)
plt.plot(1 - specificity, sensitivity, 'o', color='black')
plt.title('Testing ROC Curve')
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
plt.legend()
plt.show()

plt.figure(figsize=(6, 6), dpi=300)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
fpr, tpr, threshold = roc_curve(Y_test, Y_CT_Rad_test)
plt.plot(fpr, tpr, color="green", alpha=0.5, lw=2,
         label='CT Radiomic Model (AUC = %0.2f %%)' % AUC_CT_Rad_test)
sensitivity, specificity = sensi_speci(CT_Rad_thres, Y_CT_Rad_test, Y_test)
plt.plot(1 - specificity, sensitivity, "o", color='green')
fpr, tpr, threshold = roc_curve(Y_test, Y_CT_KSTM_test)
plt.plot(fpr, tpr, color="blue", alpha=0.5, lw=2,
         label='CT KSTM Model (AUC = %0.2f %%)' % AUC_CT_KSTM_test)
sensitivity, specificity = sensi_speci(CT_KSTM_thres, Y_CT_KSTM_test, Y_test)
plt.plot(1 - specificity, sensitivity, "o", color='blue')
fpr, tpr, threshold = roc_curve(Y_test, Y_CT_test)
plt.plot(fpr, tpr, color="darkorange", alpha=0.5, lw=2,
         label='CT Signature (AUC = %0.2f %%)' % AUC_CT_test)
sensitivity, specificity = sensi_speci(CT_thres, Y_CT_test, Y_test)
plt.plot(1 - specificity, sensitivity, "o", color='darkorange')
plt.title('Testing ROC Curve')
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
plt.legend()
plt.show()

#NRI
Y_Cl_train_num = Y_Cl_train.copy()
Y_Cl_train_num[Y_Cl_train_num >= Cl_thres] = 1
Y_Cl_train_num[Y_Cl_train_num < Cl_thres] = 0
Y_Cl_test_num = Y_Cl_test.copy()
Y_Cl_test_num[Y_Cl_test_num >= Cl_thres] = 1
Y_Cl_test_num[Y_Cl_test_num < Cl_thres] = 0

Y_cb_train_num = Y_cb_train.copy()
Y_cb_train_num[Y_cb_train_num >= cb_thres] = 1
Y_cb_train_num[Y_cb_train_num < cb_thres] = 0
Y_cb_test_num = Y_combine_test.copy()
Y_cb_test_num[Y_cb_test_num >= cb_thres] = 1
Y_cb_test_num[Y_cb_test_num < cb_thres] = 0

posi_Y_Cl_train = Y_Cl_train_num[Y_train == 1]
posi_Y_Cl_test = Y_Cl_test_num[Y_test == 1]
nega_Y_Cl_train = Y_Cl_train_num[Y_train == 0]
nega_Y_Cl_test = Y_Cl_test_num[Y_test == 0]

posi_Y_cb_train = Y_cb_train_num[Y_train == 1]
posi_Y_cb_test = Y_cb_test_num[Y_test == 1]
nega_Y_cb_train = Y_cb_train_num[Y_train == 0]
nega_Y_cb_test = Y_cb_test_num[Y_test == 0]

posi_Y_Hu_train = Y_Hu_train[Y_train == 1]
posi_Y_Hu_test = Y_Hu_test[Y_test == 1]
nega_Y_Hu_train = Y_Hu_train[Y_train == 0]
nega_Y_Hu_test = Y_Hu_test[Y_test == 0]


posi_matrix_train = metrics.confusion_matrix(posi_Y_Cl_train, posi_Y_cb_train)
posi_matrix_test = metrics.confusion_matrix(posi_Y_Cl_test, posi_Y_cb_test)
nega_matrix_train = metrics.confusion_matrix(nega_Y_Cl_train, nega_Y_cb_train)
nega_matrix_test = metrics.confusion_matrix(nega_Y_Cl_test, nega_Y_cb_test)

print(posi_matrix_train)
print(nega_matrix_train)
print(posi_matrix_test)
print(nega_matrix_test)

NRI_train = (posi_matrix_train[0][1] - posi_matrix_train[1][0])/np.sum(posi_matrix_train) + (nega_matrix_train[1][0] - nega_matrix_train[0][1])/np.sum(nega_matrix_train)
NRI_test = (posi_matrix_test[0][1] - posi_matrix_test[1][0])/np.sum(posi_matrix_test) + (nega_matrix_test[1][0] - nega_matrix_test[0][1])/np.sum(nega_matrix_test)

z_train = NRI_train/np.sqrt((posi_matrix_train[0][1] + posi_matrix_train[1][0])/np.sum(posi_matrix_train)**2 + (nega_matrix_train[1][0] + nega_matrix_train[0][1])/np.sum(nega_matrix_train)**2)
z_test = NRI_test/np.sqrt((posi_matrix_test[0][1] + posi_matrix_test[1][0])/np.sum(posi_matrix_test)**2 + (nega_matrix_test[1][0] + nega_matrix_test[0][1])/np.sum(nega_matrix_test)**2)

print('NRI: ', NRI_train, z_train, round((1-st.norm.cdf(z_train))*2, 3))
print('NRI: ', NRI_test, z_test, round((1-st.norm.cdf(z_test))*2, 3))

posi_matrix_train = metrics.confusion_matrix(posi_Y_Hu_train, posi_Y_cb_train)
posi_matrix_test = metrics.confusion_matrix(posi_Y_Hu_test, posi_Y_cb_test)
nega_matrix_train = metrics.confusion_matrix(nega_Y_Hu_train, nega_Y_cb_train)
nega_matrix_test = metrics.confusion_matrix(nega_Y_Hu_test, nega_Y_cb_test)

print(posi_matrix_train)
print(nega_matrix_train)
print(posi_matrix_test)
print(nega_matrix_test)

NRI_train = (posi_matrix_train[0][1] - posi_matrix_train[1][0])/np.sum(posi_matrix_train) + (nega_matrix_train[1][0] - nega_matrix_train[0][1])/np.sum(nega_matrix_train)
NRI_test = (posi_matrix_test[0][1] - posi_matrix_test[1][0])/np.sum(posi_matrix_test) + (nega_matrix_test[1][0] - nega_matrix_test[0][1])/np.sum(nega_matrix_test)

z_train = NRI_train/np.sqrt((posi_matrix_train[0][1] + posi_matrix_train[1][0])/np.sum(posi_matrix_train)**2 + (nega_matrix_train[1][0] + nega_matrix_train[0][1])/np.sum(nega_matrix_train)**2)
z_test = NRI_test/np.sqrt((posi_matrix_test[0][1] + posi_matrix_test[1][0])/np.sum(posi_matrix_test)**2 + (nega_matrix_test[1][0] + nega_matrix_test[0][1])/np.sum(nega_matrix_test)**2)

print(NRI_train, z_train, round((1-st.norm.cdf(z_train))*2, 3))
print(NRI_test, z_test, round((1-st.norm.cdf(z_test))*2, 3))


def dca_curve(y_true, y_score, threshold):
    num = []
    dca = []
    for t in range(len(threshold)):
        n = 0
        for j in range(len(y_score)):
            if y_score[j] < threshold[t]:
                n += 1
        num.append(n / len(y_score))
    num = np.linspace(0, 1, 500)
    for t in range(len(num)):
        Y_pred = [0] * len(y_score)
        for proba in range(len(y_score)):
            if y_score[proba] >= num[t]:
                Y_pred[proba] = 1
            else:
                Y_pred[proba] = 0
        matrix = metrics.confusion_matrix(y_true, Y_pred)
        d = matrix[1][1] / (len(y_score) + 0.0001) - matrix[0][1] / (len(y_score) + 0.0001) * num[t] / (1 - num[t])
        dca.append(d)
    return dca, num

dca_list1 = [list(np.linspace(0,1,500)),]
plt.figure(figsize=(7,5), dpi=300)
for Pb, label in [
    [Y_Hu_train, 'Experts\' Diagnoses'],
    [Y_Cl_train, 'Clinical Nomogram'],
    [Y_CT_train, 'CT Signature'],
    [Y_PET_train, 'PET Signature'],
    [Y_combine_train, 'MMF Model']
]:
    fpr, tpr, threshold = roc_curve(Y_train, Pb)
    dca, num = dca_curve(Y_train, Pb, threshold)
    dca_list1.append(dca)
    if label == 'Experts\' Diagnoses':
        plt.plot(num, dca, label=label, color='green', alpha=0.6)
    if label == 'Clinical Nomogram':
        plt.plot(num, dca, label=label, color='blue', alpha=0.6)
    if label == 'CT Signature':
        plt.plot(num, dca, label=label, color='darkorange', alpha=0.6)
    if label == 'PET Signature':
        plt.plot(num, dca, label=label, color='red', alpha=0.6)
    if label == 'MMF Model':
        plt.plot(num, dca, label=label, color='black', alpha=0.6)
dca = [0] * 500
yang = len(Y_train[Y_train == 1])
yin = len(Y_train[Y_train == 0])
pri = True
pir = True
for i in range(500):
    p = np.linspace(0, 1, 500)[i]
    dca[i] = yang / len(Y_train) - yin / len(Y_train) * p / (1 - p)
plt.plot(np.linspace(0, 1, 500), dca, linestyle='--', label='All-laparoscopy Scheme')
plt.plot([0, 1], [0, 0], linestyle='--', label='None-laparoscopy Scheme')
plt.ylim([-0.19, 0.45])
plt.ylabel('Net Benefit')
plt.xlabel('Threshold Probability')
plt.title("Training Decision Curve")
plt.legend()
plt.show()

dca_list2 = [list(np.linspace(0,1,500)),]
plt.figure(figsize=(7,5), dpi=300)
for Pb, label in [
    [Y_Hu_test, 'Experts\' Diagnoses'],
    [Y_Cl_test, 'Clinical Nomogram'],
    [Y_CT_test, 'CT Signature'],
    [Y_PET_test, 'PET Signature'],
    [Y_combine_test, 'MMF Model']
]:
    fpr, tpr, threshold = roc_curve(Y_test, Pb)
    dca, num = dca_curve(Y_test, Pb, threshold)
    dca_list2.append(dca)
    if label == 'Experts\' Diagnoses':
        plt.plot(num, dca, label=label, color='green', alpha=0.6)
    if label == 'Clinical Nomogram':
        plt.plot(num, dca, label=label, color='blue', alpha=0.6)
    if label == 'CT Signature':
        plt.plot(num, dca, label=label, color='darkorange', alpha=0.6)
    if label == 'PET Signature':
        plt.plot(num, dca, label=label, color='red', alpha=0.6)
    if label == 'MMF Model':
        plt.plot(num, dca, label=label, color='black', alpha=0.6)
dca = [0] * 500
yang = len(Y_test[Y_test == 1])
yin = len(Y_test[Y_test == 0])
pri = True
pir = True
for i in range(500):
    p = np.linspace(0, 1, 500)[i]
    dca[i] = yang / len(Y_test) - yin / len(Y_test) * p / (1 - p)
plt.plot(np.linspace(0, 1, 500), dca, linestyle='--', label='All-laparoscopy Scheme')
plt.plot([0, 1], [0, 0], linestyle='--', label='None-laparoscopy Scheme')
plt.ylim([-0.19, 0.45])
plt.ylabel('Net Benefit')
plt.xlabel('Threshold Probability')
plt.title("Testing Decision Curve")
plt.legend()
plt.show()
