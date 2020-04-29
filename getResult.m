x = [];
y = [];

load Subject_6.mat
savefile = 'Subject_6_Predictions.mat';
x=cat(3,x,X_EEG_TEST(:,201:end,:));


win_length=50;% the number of samples in one sliding window
[time_series1] = time_reshape(win_length,x,fs,y);
time_input = (time_series1 * pca_coeT(:,1:500))';
num = int16(size(time_input,2)/size(x,3));
time_input = reshape(time_input, size(time_input,1)*num , size(x,3));
time_input = (time_input' * pca_coeT2(:,1:50))';
%norm_time1=normalize(time_input',1);
norm_time1 = mapminmax('apply', time_input,ps_t);
norm_time1 = norm_time1';

Y_EEG_TEST = predict(model_svm,norm_time1);



save(savefile,'Y_EEG_TEST');
