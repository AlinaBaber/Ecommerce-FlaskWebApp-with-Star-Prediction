def results(self, target_test, predicted_test, ModelName, labels):
    target_names = labels
    print(classification_report(target_test, predicted_test, target_names=target_names))
    y_test = target_test
    preds = predicted_test
    rms = np.sqrt(np.mean(np.power((np.array(y_test) - np.array(preds)), 2)))
    score = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    pearson_coef, p_value = stats.pearsonr(y_test, preds)

    print("root mean square:", rms)
    print("score:", score)
    print("mean absolute error:", mae)
    print("mean squared error:", mse)
    print("pearson_coef:", pearson_coef)
    print("p_value:", p_value)
    print("=======================================================================\n\n")
    skplt.metrics.plot_confusion_matrix(
        y_test,
        preds,
        figsize=(10, 6), title="Confusion matrix\n Deposite Category " + ModelName)
    plt.xlim(-0.5, len(np.unique(y_test)) - 0.5)
    plt.ylim(len(np.unique(y_test)) - 0.5, -0.5)
    plt.savefig('cvroc.png')
    plt.show()
    # Bagging
    ns_probs = [0 for _ in range(len(y_test))]
    # lr_probs = predictions
    # best_found_fina=individual_list
    ns_aucb = roc_auc_score(y_test, ns_probs)
    lr_aucb = roc_auc_score(y_test, preds)
    precision, recall, thresholds = precision_recall_curve(y_test, preds)
    no_skill = len(y_test[y_test == 1]) / len(y_test)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.plot(precision, recall, thresholds, marker='.', label='Logistic')
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()
    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_aucb))
    print('Logistic: ROC AUC=%.3f' % (lr_aucb))
    # calculate roc curves
    ns_fprb, ns_tprb, _ = roc_curve(y_test, ns_probs)
    lr_fprb, lr_tprb, _ = roc_curve(y_test, preds)
    # plot the roc curve for the model
    plt.plot(ns_fprb, ns_tprb, linestyle='--', label='No Skill')
    plt.plot(lr_fprb, lr_tprb, marker='.', label='target')
    plt.plot(lr_fprb, lr_tprb, marker='.', label=ModelName)
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    plt.title("ROC Curve Graph")
    plt.savefig('comparisonroc.png')
    # show the plot
    plt.show()