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
%shuffle the training data
idx = randperm(size(x, 3));
x = x(:,:,idx);
y = y(idx,:);


%% split the data and extract the features from data

start_num = 400;
mode=1;% different modes for feature extraction
% extract the train data features and save it
win_length=50;% the number of samples in one sliding window
[time_series1] = time_reshape(win_length,x,fs,y);
pca_coeT = pca(time_series1);
time_input = (time_series1 * pca_coeT(:,1:500))';
num = int16(size(time_input,2)/size(x,3));
time_input = reshape(time_input, size(time_input,1)*num , size(x,3));
pca_coeT2 = pca(time_input');
time_input = (time_input' * pca_coeT2(:,1:50))';
%norm_time1=normalize(time_input',1);
[norm_time1(:,1:start_num),ps_t]=mapminmax(time_input(:,1:start_num));
norm_time1(:,start_num+1:size(x,3)) = mapminmax('apply', time_input(:,start_num+1:size(x,3)),ps_t);
norm_time1 = norm_time1';
% 
% freq_series1 = freq_reshape(win_length,x,fs,y);
% pca_coeF = pca(freq_series1);
% %freq_input = (freq_series1 * pca_coeF(:,1:3))';
% freq_input = freq_series1';
% freq_input = reshape(freq_input, size(freq_input,1)*size(x,1) , size(x,3));
% pca_coeF = pca(freq_input');
% freq_input = (freq_input' * pca_coeF(:,1:50))';
% [norm_gamma1(:,1:start_num),ps_f]=mapminmax(freq_input(:,1:start_num));
% norm_gamma1(:,start_num+1:size(x,3)) = mapminmax('apply', freq_input(:,start_num+1:size(x,3)),ps_f);
% norm_gamma1 = norm_gamma1';
% save('train_features.mat','norm_gamma1');



% %% frequency method
% rng(1)
% model_LD_f=fitcdiscr(norm_gamma1(1:start_num,:),y(1:start_num),'OptimizeHyperparameters',...
%     'auto','Gamma',0.8,'HyperparameterOptimizationOptions',...
%     struct('AcquisitionFunctionName','expected-improvement-plus',...
%     'MaxObjectiveEvaluations',50));
% 
% 
% % five-fold cross-validation for SVM optimization
% rng(1)
% % y(y==0)=-1;
% model_svm_f=fitcsvm(norm_gamma1(1:start_num,:),y(1:start_num),'OptimizeHyperparameters','auto',...
%     'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
%     'expected-improvement-plus','MaxObjectiveEvaluations',50));




%% evaluate time methods
rng(1)
model_LD=fitcdiscr(norm_time1(1:start_num,:),y(1:start_num),'OptimizeHyperparameters',...
    'auto','Gamma',0.8,'HyperparameterOptimizationOptions',...
    struct('AcquisitionFunctionName','expected-improvement-plus',...
    'MaxObjectiveEvaluations',50));

rng(1)
% y(y==0)=-1;
model_svm=fitcsvm(norm_time1(1:start_num,:),y(1:start_num),'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
    'expected-improvement-plus','MaxObjectiveEvaluations',50));

% ens = fitcensemble(norm_time1(1:start_num,:),y(1:start_num),'Method','Bag', ...
%    'NumLearningCycles',50,'Learners','discriminant');

y_predict(:,1) = predict(model_LD,norm_time1(start_num+1:end,:));
y_predict(:,2) = predict(model_svm,norm_time1(start_num+1:end,:));


% %% neural network
% load norm_time_all;
% checkpointPath = './';
% net=newff((norm_time1(1:start_num,:))',(y(1:start_num))',[size(norm_time1,2) 30 6 2],{'purelin','logsig','logsig','softmax'},'trainbr');
% % 10轮回显示一次结果
% net.trainParam.show=10;
% % 学习速度为0.05
% net.trainParam.lr=0.05; 
% % 最大训练次数为5000次
% net.trainParam.epochs=7000;
% % 均方误差
% net.trainParam.goal=0.65*10^(-3);  
% % 网络误差如果连续6次迭代都没有变化，训练将会自动终止（系统默认的）
% % 为了让程序继续运行，用以下命令取消这条设置
% net.divideFcn = '';
% net.trainFcn = 'traingdx';
% net.performFcn='msereg';
% net.performParam.ratio=0.5;
% options = trainingOptions('sgdm', ...
%     'InitialLearnRate',0.1, ...
%     'MaxEpochs',1000, ...
%     'Verbose',false, ...
%     'Plots','training-progress', ...
%     'Shuffle','every-epoch', ...
%     'CheckpointPath',checkpointPath);
% % 开始训练，其中pn,tn分别为输入输出样本
% net=train(net,(norm_time1(1:start_num,:))',(y(1:start_num,:))');                   
% % 利用训练好的网络，基于原始数据对BP网络仿真
% y_predict3=sim(net,(norm_time1(start_num+1:end,:))');
% y_predict3(y_predict3>=0.5) = 1;
% y_predict3(y_predict3<0.5) = 0;
%%
% y_predict(:,3) = predict(model_LD_f,norm_time1(start_num+1:end,:));
% y_predict(:,4) = predict(model_svm_f,norm_time1(start_num+1:end,:));
% 
% 
y_test = y(start_num+1:end);
y_predict_final = mean(y_predict,2);
y_predict_final = (y_predict_final>=0.5);
acc1 = 1 - sum(abs(y_predict(:,1)-y_test))/length(y_test)
acc1 = 1 - sum(abs(y_predict(:,2)-y_test))/length(y_test)
acc = 1 - sum(abs(y_predict_final-y_test))/length(y_test)
y_predict(:,4) = y_test;
y_predict(:,5) = y_predict_final;

fprintf('The accuracy of Time domain LD through 5-fold cross-validation is %.2f %% \n',...
    (1-model_LD.HyperparameterOptimizationResults.MinObjective))

fprintf('The accuracy of SVM through 5-fold cross-validation is %.2f %% \n',...
    (1-model_svm.HyperparameterOptimizationResults.MinObjective))




% model_name = ['LD','SVM','NN','Ensemble'];
% figure()
% result = [0.71,0.73; 0.72,0.75;0.81, 0.65];
% bar(result)
% 
% ylim([0,1])
% set(gca,'xtick',[1 2 3 4])
% set(gca,'xticklabel',{'LD','SVM','NN'})
% title('Accuracy of three different models')
% legend('result on training set','result on validation set')
% xlabel('Models')
% ylabel('Accuracy')


