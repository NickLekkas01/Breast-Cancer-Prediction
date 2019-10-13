filename = 'breast-cancer-wisconsin.data';
filename2 = 'mhx.xlsx';
col = textread(filename, '%s');
K = strsplit(string(col(1)), ',');
for i=2:699
    str = string(col(i));
    B = strsplit(str, ',');
    T = K;
    K = [T; B];
end
%Most found value of each column
Mode = mode(K);
%Replace '?' with most found value
for i=1:11
    lid = K(:, i) == '?';
    K(lid,i) = Mode(i);
end
%Make string to a numberic value
K = double(K);
%Write to excel
xlswrite(filename2, K);

%Delete ID column from matrix
K(:,1) = [];
%Keep data classes
data_classes = K(:,10);
%Delete class column from matrix
K(:,10) = [];

%Normalization
K = normalize(K, 'range') ;
%data is the remaining matrix
data = K;

%-----------------KNN------------------------
knn_data         = data;
knn_data_classes = data_classes;

%Generate cross-validation indices
knn_indices = crossvalind('Kfold',knn_data_classes,10);

cp_knn_5  = classperf(knn_data_classes);
cp_knn_10 = classperf(knn_data_classes);

%------------------------%

for i = 1:10
    knn_test = (knn_indices == i); 
    knn_train = ~knn_test;
    mdl_knn_5 = fitcknn(knn_data(knn_train,:),knn_data_classes(knn_train),'NumNeighbors',5);
    knn_predictions = predict(mdl_knn_5,knn_data(knn_test,:));
    cp1 = classperf(cp_knn_5 , knn_predictions , knn_test);
end
fprintf("KNN 5 Neighbors\n");
cp1

%------------------------%

for i = 1:10
    knn_test = (knn_indices == i); 
    knn_train = ~knn_test;
    mdl_knn_10 = fitcknn(knn_data(knn_train,:),knn_data_classes(knn_train),'NumNeighbors',10);
    knn_predictions = predict(mdl_knn_10,knn_data(knn_test,:));
    cp2 = classperf(cp_knn_10 , knn_predictions , knn_test);
end
fprintf("KNN 10 Neighbors\n");
cp2

%-----------------Naive Bayes------------------------
nb_data         = data;
nb_data_classes = data_classes;

%Generate cross-validation indices
nb_indices = crossvalind('Kfold',nb_data_classes,10);

cp_nb1  = classperf(nb_data_classes);
cp_nb2 = classperf(nb_data_classes);

%Frequency table
tabulate(nb_data_classes);  
%------------------------%
prior = [0.3 0.7];
for i = 1:10
    nb_test = (nb_indices == i); 
    nb_train = ~knn_test;
    mdl_nb = fitcnb(nb_data(nb_train,:),nb_data_classes(nb_train),'Prior',prior);
    nb_predictions = predict(mdl_nb,nb_data(nb_test,:));
    cp1 = classperf(cp_nb1 , nb_predictions , nb_test);
end
fprintf("Bayes 0.3 0.7\n");
cp1

%------------------------%

prior = [0.2 0.8];
for i = 1:10
    nb_test = (nb_indices == i); 
    nb_train = ~knn_test;
    mdl_nb = fitcnb(nb_data(nb_train,:),nb_data_classes(nb_train),'Prior',prior);
    nb_predictions = predict(mdl_nb,nb_data(nb_test,:));
    cp2 = classperf(cp_nb2 , nb_predictions , nb_test);
end
fprintf("Bayes 0.2 0.8\n");
cp2

%-----------------SVM------------------------
svm_data         = data;
svm_data_classes = data_classes;

%Generate cross-validation indices
svm_indices = crossvalind('Kfold',svm_data_classes,10);

cp_svm1  = classperf(svm_data_classes);
cp_svm2 = classperf(svm_data_classes);

%------------------------%

for i = 1:10
    svm_test = (svm_indices == i); 
    svm_train = ~svm_test;
    mdl_svm = fitcsvm(svm_data(svm_train,:),svm_data_classes(svm_train),'Standardize',true,'KernelFunction','RBF',...
    'KernelScale','auto');
    svm_predictions = predict(mdl_svm,svm_data(svm_test,:));
    cp1 = classperf(cp_svm1 , svm_predictions , svm_test);
end
fprintf("SVM RBF\n");
cp1

%------------------------%

for i = 1:10
    svm_test = (svm_indices == i); 
    svm_train = ~svm_test;
    mdl_svm = fitcsvm(svm_data(svm_train,:),svm_data_classes(svm_train),'Standardize',true,'KernelFunction','polynomial',...
    'KernelScale','auto');
    svm_predictions = predict(mdl_svm,svm_data(svm_test,:));
    cp2 = classperf(cp_svm2 , svm_predictions , svm_test);
end
fprintf("SVM polynomial\n");
cp2

%-----------------Decision Tree------------------------
DT_data         = data;
DT_data_classes = data_classes;

%Generate cross-validation indices
DT_indices = crossvalind('Kfold',DT_data_classes,10);

cp_DT1  = classperf(DT_data_classes);
cp_DT2 = classperf(DT_data_classes);

%------------------------%

for i = 1:10
    DT_test = (DT_indices == i); 
    DT_train = ~DT_test;
    mdl_DT = fitctree(DT_data(DT_train,:),DT_data_classes(DT_train));
    DT_predictions = predict(mdl_DT,DT_data(DT_test,:));
    cp1 = classperf(cp_DT1 , DT_predictions , DT_test);
end
fprintf("Decision Tree MaxNumSplits Default\n");
cp1
%------------------------%

for i = 1:10
    DT_test = (DT_indices == i); 
    DT_train = ~DT_test;
    mdl_DT = fitctree(DT_data(DT_train,:),DT_data_classes(DT_train), 'MaxNumSplits',7);
    DT_predictions = predict(mdl_DT,DT_data(DT_test,:));
    cp2 = classperf(cp_DT2 , DT_predictions , DT_test);
end
fprintf("Decision Tree MaxNumSplits 7\n");
cp2