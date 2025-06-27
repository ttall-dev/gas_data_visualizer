% test moyenne mobile
close all
clear
N=2^8;
f = (0:1/N:1-1/N)-1/2;
function [f,spectre] = tfFiltreLissage(order)
N=2^8;
f = (0:1/N:1-1/N)-1/2;
filter = zeros(size(f));
for i=0:order-1
    filter = filter + exp(-1i*2*pi*f*i);
end
spectre = filter;
end

%% Filtre moyenneur 
figure; 
hold on
maxOrder = 4;
minOrder = 0;

% Initialize a cell array to hold the legend labels
legendLabels = cell(maxOrder - minOrder + 1, 1);

for order = minOrder:maxOrder
    [f, filter] = tfFiltreLissage(order);
    plot(f, abs(filter));
    
    % Store the label for the current order
    legendLabels{order + 1} = sprintf('Order %d', order);
end

grid("on")
xlabel("Normalized frequency")
ylabel("FFT")
title("Averaging Filter - Freq Response Analysis")

% Add the legend with the labels
legend(legendLabels, 'Location', 'best');

figure; 
hold on

% Initialize a cell array to hold the legend labels
legendLabels = {};


% f = linspace(0, 1, 100); 

for a = 0:0.2:1
    wightedFilterTF = a/4 * exp(2 * 1i * pi * f * 2) + ...
                      a * exp(2 * 1i * pi * f) + ...
                      1 + ...
                      a * exp(2 * 1i * pi * f) + ...
                      a/4 * exp(2 * 1i * pi * f * 2);
    plot(f, abs(wightedFilterTF));  % Use abs() to plot the magnitude
    
    % Store the label for the current value of 'a'
    legendLabels{end + 1} = sprintf('a = %.1f', a);
end

grid("on")
xlabel("Normalized frequency")
ylabel("FFT")
title("Averaging Filter - Freq Response Analysis")

% Add the legend with the labels
legend(legendLabels, 'Location', 'best');

%% filtre moyenneur adaptable

% avgFilter = (1./(-1:0.25:1)).^8;
avgFilter = hamming(8);

avgFilter(ceil(end/2))=avgFilter(floor(end/2));
fftAvgFilter = fftshift(fft(avgFilter,256));

figure,
subplot(2,1,1)
stem(avgFilter)
xlabel("Samples")
ylabel("Averaging Filter (Hamming)")
subplot(2,1,2)
f= linspace(-1/2,1/2,length(fftAvgFilter));
plot(f,abs(fftAvgFilter))
ylabel("FFT")
grid("on")
xlabel("Normalized frequency")
sgtitle("Averaging Filter - Freq & Time Response Analysis")

%% Reverse synthesis:

fs = 512;
Nfft = 512;
f = linspace(-1/2,1/2,Nfft);

% target = [15, 200]/fs; % Hz
f0 = 25/fs;
f1 = 125/fs;

bpf = zeros(Nfft,1);

bpf(f<f1)=1;
bpf(f<f0)=0;
figure
subplot(2,1,1)
plot(f,bpf)
grid("on")
xlabel("Normalized frequency")
ylabel("FFT")
title("Averaging Filter - Freq Response Analysis")

subplot(2,1,2)
plot(abs(fftshift(ifft(bpf))))



%% Synthèse: sinc(t) -> porte(f)

theta = (f1-f0)*fs;
t = 0:1/1000:1-1/fs;
fc = (f0+f1)/2*fs;
k0=35;
h = theta * exp(2*1i*pi*fc*(t-k0/theta)).*sin(theta*pi*(t-k0/theta))./(theta*pi*(t-k0/theta));

figure
subplot(2,1,1)
plot(t,abs(h))
xlabel("t")
ylabel("Synthesized filter")
Nfft = 1024;
spectre = abs(fftshift(fft(h,Nfft)));
subplot(2,1,2)
hold on
plot(linspace(-1/2,1/2,Nfft),spectre)
plot([f0 f1],[0 0],'or')


