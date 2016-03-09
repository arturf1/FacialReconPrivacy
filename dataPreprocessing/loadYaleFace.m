function loadYaleFace()

directory_name = 'yalefaces';
files = dir(directory_name);
fileIndex = find(~[files.isdir]);

DATA = [];

    for i = 1:length(fileIndex)
        
        A = imread(strcat('yalefaces/', files(fileIndex(i)).name)); 

        DATA = cat(3, DATA, A);
        
    end

    size(DATA)
    assignin('base', 'DATA', DATA);

save('yalefaces.mat', 'DATA');
clear;
end