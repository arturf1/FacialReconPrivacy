function SpiesFE(database)

load(database);

DATA = evalin('base','DATA');
SpiesFM = [];

R = [];
I = [];

for i = 1:size(DATA, 3)
    A = DATA(:,:,i);
    
    % apply FFT
    FV = fft2(A);    
    FV=fftshift(FV);
    

    I = cat(3, I, imag(FV));
    R = cat(3, R, real(FV));
end

% compute frequency variance
R_var = var(R,0,3);
I_var = var(I,0,3);

% select most varied frequencies
numRealFreq = 22;
[value,index]=sort(R_var(:), 'descend');
maskReal = zeros(size(R_var));

for i = 1:numRealFreq
    [row,column] = ind2sub(size(R_var), index(i));
    maskReal(row,column) = 1;
end
%figure,imshow(log(R_var),[]);
%figure,imshow(maskReal,[]);

numImgFreq = 8;
[value,index]=sort(I_var(:), 'descend');
maskImg = zeros(size(I_var));

for i = 1:numImgFreq
    [row,column] = ind2sub(size(I_var), index(i));
    maskImg(row,column) = 1;
end
%figure,imshow(log(I_var),[]);
%figure,imshow(maskImg,[]);


for i = 1:size(DATA, 3)   
    C = I(:,:,i).*maskImg;
    D = R(:,:,i).*maskReal;
    
    % find all none zero frequencies
    [row,col,c] = find(C);
    [row,col,d] = find(D);
    
    FV = [d;c];

    SpiesFM = [SpiesFM FV(:)]; 
end

size(SpiesFM)
assignin('base', 'SpiesFM', SpiesFM);
save(strcat('SpiesFM', database), 'SpiesFM');

clear;
end