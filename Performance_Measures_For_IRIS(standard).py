

#09-02-2021, Pintu
#181CO139
#Our aim is to build the function for calculating the confusion_matrix and classification_report
# for multiclass classification, like IRIS dataset.

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

#Function for confusion matrix.
def Confusion_matrix(y_test, y_pred, target_names=None):
    #target_names is a list.
    #actual values are arranged in the rows.
    # predicted values are arranged in the columns.
    #if there are m classes, then cm is m*m matrix.
    m=len(target_names)
    size = len(y_test)
    matrix=dict()

    #create matrix initialised with 0
    for class_name in range(m):
        matrix[class_name]=[0 for k in range(m)]

    #populating the matrix.
    for i in range(size):
        actual_class=y_test[i]
        pred_class = y_pred[i]
        matrix[actual_class][pred_class]+=1

    #Change the name of columns.
    if target_names==None:
        pass
    else:
        matrix=dict(zip(target_names,list(matrix.values())))

    #Now, lets print the confusion matrix.
    print("Confusion Matrix of given model is :")
    print("Count=%-14d %-15s %-15s %-15s"%(size,target_names[0], target_names[1], target_names[2]))
    for key,value in matrix.items():
        print("Actual %-13s %-15d %-15d %-15d"%(key,value[0],value[1],value[2]))

    return matrix

#Function for performance report.
def performance_report(cm):
    col= len(cm)
    #col=number of class
    arr=[]
    for key,value in cm.items():
        arr.append(value)

    cr=dict()
    support_sum=0
    macro=[0]*4
    weighted=[0]*4
    for i in range(col):
        horizontal_sum= sum([arr[j][i] for j in range(col)])
        vertical_sum= sum(arr[i])
        p = arr[i][i] / horizontal_sum
        r = arr[i][i] / vertical_sum
        f = (2 * p * r) / (p + r)
        s = vertical_sum
        row=[p,r,f,s]
        support_sum+=s
        for j in range(4):
            macro[j]+=row[j]
            weighted[j]+=row[j]*s
        cr[i]=row

    #add Accuracy parameters.
    truepos=0
    total=0
    for i in range(col):
        truepos+=arr[i][i]
        total+=sum(arr[i])

    cr['Accuracy']=["", "", truepos/total, support_sum]

    #Add macro-weight and weighted_avg features.
    macro_avg=[Sum/col for Sum in macro]
    cr['Macro_avg']=macro_avg

    weighted_avg=[Sum/support_sum for Sum in weighted]
    cr['Weighted_avg']=weighted_avg

    #print the classification_report
    print("Performance report of the model is :")
    space,p,r,f,s=" ","Precision","Recall","F1-Score","Support"
    print("%13s %9s %9s %9s %9s\n"%(space,p,r,f,s))
    stop=0
    for key,value in cr.items():
        if stop<col:
            stop+=1
            print("%13s %9.2f %9.2f %9.2f %9d"%(key,value[0],value[1],value[2],value[3]))
        elif stop==col:
            stop+=1
            print("\n%13s %9s %9s %9.2f %9d"%(key,value[0],value[1],value[2],value[3]))
        else:
            print("%13s %9.2f %9.2f %9.2f %9d"%(key,value[0],value[1],value[2],value[3]))


# from sklearn.metrics import confusion_matrix,classification_report
#Main Function is here.
def main():
    dataset=load_iris()
    X,y,classes=dataset['data'],dataset['target'],dataset['target_names']

    X_train,X_test,y_train,y_test=train_test_split(X,y,shuffle=True,random_state=5,test_size=0.3)
    model=GaussianNB().fit(X_train,y_train)
    y_pred=model.predict(X_test)
    classes=list(classes)
    cm=Confusion_matrix(y_test, y_pred, classes)
    cr=performance_report(cm)


if __name__ == '__main__':
    main()
