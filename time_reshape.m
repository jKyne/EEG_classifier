function [time_series] = time_reshape(win_length,x,fs,y)
    step  = 30;
    shape = size(x);
    x1 = permute(x, [2 1 3] );
    x2 = reshape(x1,shape(1)*shape(2),shape(3));
    % filter and normalize
    xFilt = eegfilt(x2',fs,0.5,40,shape(1),6);
    x_pro = reshape(xFilt',shape(2),shape(1),shape(3));
   
 
    num = 1;
    for i=1:shape(3)
         % denoise with PCA
%         [U,S,V] = svd(x_pro(:,:,i));
%         S(:,40:end) = 0;
%         x_pro(:,:,i) = U*S*V';
        for j = 0:floor((shape(2)-win_length)/step)
            time_series(:,num) = reshape(x_pro(j*step+1:j*step+win_length,:,i),shape(1)*win_length,1);
            num = num+1;
        end
    end
    time_series = time_series';
    
%     figure()
%     subplot(311)
%     plot(x(1,1:50,1))
%     subplot(312)
%     plot(xFilt(1,1:50))
%     subplot(313)
%     plot(time_series(1,1:50))
   
end