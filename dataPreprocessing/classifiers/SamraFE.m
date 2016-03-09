function SamraFE(database)

load(database);

DATA = evalin('base','DATA');
SamraFM = []

for i = 1:size(DATA, 3)
    A = DATA(:,:,i);

    % apply WT 3 times
    [cA,cH,cV,cD] = dwt2(A, 'db4');
    [ccA,ccH,ccV,ccD] = dwt2(cA, 'db4');
    [cccA,cccH,cccV,cccD] = dwt2(ccA, 'db4');
    
    % apply FFT to LL of WT
    FV = fft2(cccA);
    
    % log
    FV=fftshift(FV);
    FV=abs(FV);
    FV=log(1+FV);
    
    SamraFM = [SamraFM FV(:)];   
end

size(SamraFM)
assignin('base', 'SamraFM', SamraFM);
save(strcat('SamraFM', database), 'SamraFM');
clear;
end