def f1_score_3_class(prediction, y):
    TP1, FP1, FN1, TN1 = 0, 0, 0, 0
    TP2, FP2, FN2, TN2 = 0, 0, 0, 0
    TP3, FP3, FN3, TN3 = 0, 0, 0, 0
    for i in range(len(y)):
        Act, Pre = y[i], prediction[i]

        ## for class 1
        if Act == 0 and Pre == 0: TP1 += 1
        if Act == 0 and Pre != 0: FN1 += 1
        if Act != 0 and Pre == 0: FP1 += 1
        if Act != 0 and Pre != 0: TN1 += 1
        ## for class 2
        if Act == 1 and Pre == 1: TP2 += 1
        if Act == 1 and Pre != 1: FN2 += 1
        if Act != 1 and Pre == 1: FP2 += 1
        if Act != 1 and Pre != 1: TN2 += 1
        ## for class 3
        if Act == 2 and Pre == 2: TP3 += 1
        if Act == 2 and Pre != 2: FN3 += 1
        if Act != 2 and Pre == 2: FP3 += 1
        if Act != 2 and Pre != 2: TN3 += 1

    ## print result
    Acc_all = float(TP1 + TP2 + TP3 ) / float(len(y))
    Acc1 = float(TP1 + TN1) / float(TP1 + TN1 + FN1 + FP1)
    if (TP1 + FP1) == 0:
        Prec1 = 0
    else:
        Prec1 = float(TP1) / float(TP1 + FP1)
    if (TP1 + FN1) == 0:
        Recll1 = 0
    else:
        Recll1 = float(TP1) / float(TP1 + FN1)
    if (Prec1 + Recll1) == 0:
        F1 = 0
    else:
        F1 = 2 * Prec1 * Recll1 / (Prec1 + Recll1)

    Acc2 = float(TP2 + TN2) / float(TP2 + TN2 + FN2 + FP2)
    if (TP2 + FP2) == 0:
        Prec2 = 0
    else:
        Prec2 = float(TP2) / float(TP2 + FP2)
    if (TP2 + FN2) == 0:
        Recll2 = 0
    else:
        Recll2 = float(TP2) / float(TP2 + FN2)
    if (Prec2 + Recll2) == 0:
        F2 = 0
    else:
        F2 = 2 * Prec2 * Recll2 / (Prec2 + Recll2)

    Acc3 = float(TP3 + TN3) / float(TP3 + TN3 + FN3 + FP3)
    if (TP3 + FP3) == 0:
        Prec3 = 0
    else:
        Prec3 = float(TP3) / float(TP3 + FP3)
    if (TP3 + FN3) == 0:
        Recll3 = 0
    else:
        Recll3 = float(TP3) / float(TP3 + FN3)
    if (Prec3 + Recll3)== 0:
        F3 = 0
    else:
        F3 = 2 * Prec3 * Recll3 / (Prec3 + Recll3)

    return [F1, F2, F3]

def f1_score_4_class(prediction, y):
    TP1, FP1, FN1, TN1 = 0, 0, 0, 0
    TP2, FP2, FN2, TN2 = 0, 0, 0, 0
    TP3, FP3, FN3, TN3 = 0, 0, 0, 0
    TP4, FP4, FN4, TN4 = 0, 0, 0, 0
    for i in range(len(y)):
        Act, Pre = y[i], prediction[i]

        ## for class 1
        if Act == 0 and Pre == 0: TP1 += 1
        if Act == 0 and Pre != 0: FN1 += 1
        if Act != 0 and Pre == 0: FP1 += 1
        if Act != 0 and Pre != 0: TN1 += 1
        ## for class 2
        if Act == 1 and Pre == 1: TP2 += 1
        if Act == 1 and Pre != 1: FN2 += 1
        if Act != 1 and Pre == 1: FP2 += 1
        if Act != 1 and Pre != 1: TN2 += 1
        ## for class 3
        if Act == 2 and Pre == 2: TP3 += 1
        if Act == 2 and Pre != 2: FN3 += 1
        if Act != 2 and Pre == 2: FP3 += 1
        if Act != 2 and Pre != 2: TN3 += 1
        ## for class 4
        if Act == 3 and Pre == 3: TP4 += 1
        if Act == 3 and Pre != 3: FN4 += 1
        if Act != 3 and Pre == 3: FP4 += 1
        if Act != 3 and Pre != 3: TN4 += 1

    ## print result
    Acc_all = float(TP1 + TP2 + TP3 + TP4) / float(len(y))
    Acc1 = float(TP1 + TN1) / float(TP1 + TN1 + FN1 + FP1)
    if (TP1 + FP1) == 0:
        Prec1 = 0
    else:
        Prec1 = float(TP1) / float(TP1 + FP1)
    if (TP1 + FN1) == 0:
        Recll1 = 0
    else:
        Recll1 = float(TP1) / float(TP1 + FN1)
    if (Prec1 + Recll1) == 0:
        F1 = 0
    else:
        F1 = 2 * Prec1 * Recll1 / (Prec1 + Recll1)

    Acc2 = float(TP2 + TN2) / float(TP2 + TN2 + FN2 + FP2)
    if (TP2 + FP2) == 0:
        Prec2 = 0
    else:
        Prec2 = float(TP2) / float(TP2 + FP2)
    if (TP2 + FN2) == 0:
        Recll2 = 0
    else:
        Recll2 = float(TP2) / float(TP2 + FN2)
    if (Prec2 + Recll2) == 0:
        F2 = 0
    else:
        F2 = 2 * Prec2 * Recll2 / (Prec2 + Recll2)

    Acc3 = float(TP3 + TN3) / float(TP3 + TN3 + FN3 + FP3)
    if (TP3 + FP3) == 0:
        Prec3 = 0
    else:
        Prec3 = float(TP3) / float(TP3 + FP3)
    if (TP3 + FN3 ) == 0:
        Recll3 = 0
    else:
        Recll3 = float(TP3) / float(TP3 + FN3)
    if (Prec3 + Recll3) == 0:
        F3 = 0
    else:
        F3 = 2 * Prec3 * Recll3 / (Prec3 + Recll3)

    Acc4 = float(TP4 + TN4) / float(TP4 + TN4 + FN4 + FP4)
    if (TP4 + FP4) == 0:
        Prec4 = 0
    else:
        Prec4 = float(TP4) / float(TP4 + FP4)
    if (TP4 + FN4) == 0:
        Recll4 = 0
    else:
        Recll4 = float(TP4) / float(TP4 + FN4)
    if (Prec4 + Recll4) == 0:
        F4 = 0
    else:
        F4 = 2 * Prec4 * Recll4 / (Prec4 + Recll4)

    return [F1, F2, F3, F4]