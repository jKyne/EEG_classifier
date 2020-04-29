clear;
load eeg-ern.mat
%% preprocess
% flat
x10 = permute(x1, [2 1 3] );
x20 = permute(x2, [2 1 3] );
x11 = reshape(x10,64*25,79);
x21 = reshape(x20,64*25,300);

% filter and normalize
xFilt1 = eegfilt(x11',fs,0.5,70,64,6);
xFilt2 = eegfilt(x21',fs,0.5,70,64,4);
x1_pro = xFilt1'/sqrt(norm(xFilt1));
x2_pro = xFilt2'/sqrt(norm(xFilt2));
% reshape
x1_pro = reshape(x1_pro,25,64,79);
x2_pro = reshape(x2_pro,25,64,300);
% figure()
% subplot(311)
% plot(x1(1:25,1))
% hold on; plot(x2(1:25,1))
% subplot(312)
% plot(xFilt1(1,1:25))
% hold on; plot(xFilt2(1,1:25))
% subplot(313)
% plot(x1_pro(:,1,1))
% hold on; plot(x2_pro(:,1,1))
%% LDA
% v:(64*D);x[k]:64*D; window with D
D = 25;
num = 1;
for i=1:79
    for j = 0:25-D
        x(:,num) = reshape(x1_pro(j+1:j+D,:,i),64*D,1);
        c(num) = 0;
        num = num+1;
    end
end
for i=1:300
    for j = 0:25-D
        x(:,num) = reshape(x2_pro(j+1:j+D,:,i),64*D,1);
        c(num) = 1;
        num = num+1;
        
    end
end

%% LDA
Model=fitcdiscr(x',c','DiscrimType','linear'); 
v = pinv(Model.Sigma)*(Model.Mu(1,:)-Model.Mu(2,:))';
v0 = -0.5*(Model.Mu(1,:))*pinv(Model.Sigma)*Model.Mu(1,:)'+0.5*(Model.Mu(2,:))*pinv(Model.Sigma)*Model.Mu(2,:)'+log(79/300);
figure()
subplot(121)
plot(v)
title('v')
subplot(122)
plot(v'*x+v0)
title('v^T*x')

% ROC
p_tmp = 1./(1+exp((v'*x+v0)));
K = 25-D+1;
for i=1:379
    p(i) = mean(p_tmp((i-1)*K+1:i*K));
end
c = zeros(1,379);
c(80:end) = 1;
[predict,index] = sort(p);  
ground_truth = c(index);  
  
a = 1.0;  
b = 1.0;  
pos_num = sum(ground_truth==1);  
neg_num = sum(ground_truth==0);   
x_step = 1.0/neg_num;  
y_step = 1.0/pos_num;  

for i=1:length(ground_truth)  
    if ground_truth(i) == 1  
        b = b - y_step;  
    else  
        a = a - x_step;  
    end  
    X(i)=a;  
    Y(i)=b;  
end  
%%
figure()
plot(X,Y,'-r','LineWidth',2,'MarkerSize',3);  
xlabel('False positive rate');  
ylabel('True positive rate');  
title('ROC curve');  
xlim([0,1])
ylim([0,1])
Az = -trapz(X,Y);   
text(0.5,0.5,['Az = ',num2str(Az)])

%% leave one out
for k = 1: size(x,2)
    xk = x(:,k);
    ck = c(k);
    xtmp = [x(:,1:k-1),x(:,k+1:end)];
    ctmp = [c(:,1:k-1),c(:,k+1:end)];
    Model=fitcdiscr(xtmp',ctmp','DiscrimType','linear'); 
    v = pinv(Model.Sigma)*(Model.Mu(1,:)-Model.Mu(2,:))';
    v0 = -0.5*(Model.Mu(1,:))*pinv(Model.Sigma)*Model.Mu(1,:)'+0.5*(Model.Mu(2,:))*pinv(Model.Sigma)*Model.Mu(2,:)'+log(79/300);
    p(k) = 1/(1+exp(v'*xk+v0));
end
%% ROC
[predict,index] = sort(p(1:337));  
ground_truth = c(index);  
  
a = 1.0;  
b = 1.0;  
pos_num = sum(ground_truth==1);  
neg_num = sum(ground_truth==0);   
x_step = 1.0/neg_num;  
y_step = 1.0/pos_num;  

for i=1:length(ground_truth)  
    if ground_truth(i) == 1  
        b = b - y_step;  
    else  
        a = a - x_step;  
    end  
    X(i)=a;  
    Y(i)=b;  
end  
%%
figure()
plot(X,Y,'-r','LineWidth',2,'MarkerSize',3);  
xlabel('False positive rate');  
ylabel('True positive rate');  
title('ROC curve for leave-one-out');  
xlim([0,1])
ylim([0,1])
Az = -trapz(X,Y);   
text(0.5,0.5,['Az = ',num2str(Az)])