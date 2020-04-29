function [freq_series] = freq_reshape(win_length,x,fs,y)
    step  = 30;
    shape = size(x);
    x1 = permute(x, [2 1 3] );
    
    for i=1:shape(3)
        for j = 1:shape(1)
            freq_series1(:,j,i) = Feature(x1(:,j,i),fs,1);
        end
    end
    
    freq_series = reshape(freq_series1,size(freq_series1,1),size(freq_series1,2)*size(freq_series1,3));
    freq_series = freq_series';
    
%     figure()
%     subplot(311)
%     plot(x(1,1:50,1))
%     subplot(312)
%     plot(xFilt(1,1:50))
%     subplot(313)
%     plot(time_series(1,1:50))
   
end