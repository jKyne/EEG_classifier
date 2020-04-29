clc;clear;
%% load the data
x = [];
y = [];
load Subject_1.mat
x=cat(3,x,X_EEG_TRAIN(:,201:end,:));% signal
y=[y;Y_EEG_TRAIN];% label   % Face:1 Car:0
load Subject_2.mat
x=cat(3,x,X_EEG_TRAIN(:,201:end,:));% signal
y=[y;Y_EEG_TRAIN];% label   % Face:1 Car:0
load Subject_3.mat
x=cat(3,x,X_EEG_TRAIN(:,201:end,:));% signal
y=[y;Y_EEG_TRAIN];% label   % Face:1 Car:0
load Subject_4.mat
x=cat(3,x,X_EEG_TRAIN(:,201:end,:));% signal
y=[y;Y_EEG_TRAIN];% label   % Face:1 Car:0
load Subject_5.mat
x=cat(3,x,X_EEG_TRAIN(:,201:end,:));% signal
y=[y;Y_EEG_TRAIN];% label   % Face:1 Car:0
% load Subject_6.mat
% x=cat(3,x,X_EEG_TRAIN(:,201:end,:));% signal
% y=[y;Y_EEG_TRAIN];% label   % Face:1 Car:0
load Subject_7.mat
x=cat(3,x,X_EEG_TRAIN(:,201:end,:));% signal
y=[y;Y_EEG_TRAIN];% label   % Face:1 Car:0
load Subject_8.mat
x=cat(3,x,X_EEG_TRAIN(:,201:end,:));% signal
y=[y;Y_EEG_TRAIN];% label   % Face:1 Car:0
fs=1000;

%% split the data and extract the features from data


mode=1;% different modes for feature extraction
% extract the train data features and save it
win_length=50;% the number of samples in one sliding window
fm1=split_and_feature(win_length,x,fs,mode);
gamma_band1=reshape(fm1,[size(fm1,3) size(fm1,1)*size(fm1,2)]);
norm_gamma1=normalize(gamma_band1,1);

save('train_features.mat','norm_gamma1');

% extract the test data features and save it
x_test=X_EEG_TEST;
fm2=split_and_feature(win_length,x_test,fs,mode);
gamma_band2=reshape(fm2,[size(fm2,3) size(fm2,1)*size(fm2,2)]);
norm_gamma2=normalize(gamma_band2,1);

save('test_features.mat','norm_gamma2');
%% five-fold cross-validation for Linear Discriminant optimization
rng(1)
model_LD=fitcdiscr(norm_gamma1(1:400,:),y(1:400),'OptimizeHyperparameters',...
    'auto','Gamma',0.8,'HyperparameterOptimizationOptions',...
    struct('AcquisitionFunctionName','expected-improvement-plus',...
    'MaxObjectiveEvaluations',50));


%% five-fold cross-validation for SVM optimization
rng(1)
% y(y==0)=-1;
model_svm=fitcsvm(norm_gamma1(1:400,:),y(1:400),'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
    'expected-improvement-plus','MaxObjectiveEvaluations',50));

%% ensenble learning for KNN
ens = fitcensemble(norm_gamma1,y,'Method','Subspace', ...
   'NumLearningCycles',50,'Learners','KNN');

%% the accuracy display after 100 iterations optimization
fprintf('The accuracy of LD through 5-fold cross-validation is %.2f %% \n',...
    (1-model_LD.HyperparameterOptimizationResults.MinObjective))

fprintf('The accuracy of SVM through 5-fold cross-validation is %.2f %% \n',...
    (1-model_svm.HyperparameterOptimizationResults.MinObjective))

fprintf('The accuracy of ensemble KNN through 5-fold cross-validation is %.2f %% \n',...
    (1-ens.HyperparameterOptimizationResults.MinObjective))
%% test the model performance
%accuracy=cross_val(gamma_band1,y);
%fprintf('The accuracy of the model in this dataset is: %.2f %% ',accuracy*100)

y1 = predict(model_LD,norm_gamma1(401:end,:));
y2 = predict(model_svm,norm_gamma1(401:end,:));
y_test = y(401:end);
loss1 = sum(abs(y1-y_test))/length(y_test)
loss2 = sum(abs(y2-y_test))/length(y_test)




















