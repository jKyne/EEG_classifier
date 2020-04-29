function [new_feature]=split_and_feature(win_length,sample,fs,mode)
% win_length: the length of the sliding window
% sample: the input EEG data with different channels
% fs: the sampling frequency
% resample: splitted sapmles

win_num=floor(size(sample,2)./win_length);
s={};

new_feature=[];

for sub=1:size(sample,3)% subjects
    
for i=1:size(sample,1)% channels
    
    for j=1:win_num  % splitting the data
       
        s{j}=sample(i,(j-1)*win_length+1:j*win_length,sub);
        value=Feature(s{j},fs,mode);
        new_feature(i,j,sub)=value;
    end
    
end
    
end






end