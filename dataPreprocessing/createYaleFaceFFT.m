function createYaleFaceFFT()

directory_name = 'yalefaces';
files = dir(directory_name);
fileIndex = find(~[files.isdir]);

DATA = [];

    for i = 1:size(fileIndex,2)
        
        A = imread(strcat('yalefaces/', files(fileIndex(i)).name)); 
        % make the number of columns odd
        A = A(:, 1:end-1);
           
        FV = fft2(A);
        FV = fftshift(FV);

        % For all REAL (as opposed to IMAGINARY or COMPLEX) images, the FT is 
        % symmetrical about the origin so the 1st and 3rd quadrants are the 
        % same and the 2nd and 4th quadrants are the same.
        FV = FV(1:122,:);
    
        R = abs(FV);
        theta = angle(FV);
        
        FV = [R(:); theta(:)];

        DATA = [DATA FV(:)]; 
    end

    size(DATA)
    assignin('base', 'DATA', DATA);

save('yalefacesFFT.mat', 'DATA');
clear;
end



