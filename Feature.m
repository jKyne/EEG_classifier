function [value]=Feature(sample,fs,mode)
% sample: the splitted samples
% fs: the sampling frequency
% gamma_value: the average power in the specific gamma band
% mode: 0: gamma band: 25-75 Hz;   1: gamma band: 30-100 Hz 2: whole
% frequency band: theta: 3-7 Hz alpha: 8-12 Hz beta: 13-29 Hz gamma: 30-100 Hz

FFT1=fft(sample);

f=linspace(0,fs./2-1,length(FFT1)./2);
fft1=FFT1(1:length(FFT1)./2);

if mode==0
    
    f1=30;
    f2=100;
    ind=(f>=f1&f<=f2);
    f_gamma=fft1(ind);
    gamma_value=mean(power(abs(f_gamma),2));
end

if mode==1
    
    f1=0.5;
    f2=40;
    ind=(f>=f1&f<=f2);
    f_gamma=fft1(ind);
    gamma_value=power(abs(f_gamma),2);
    
end


value=gamma_value;
end