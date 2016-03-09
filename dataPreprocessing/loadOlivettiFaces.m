function loadOlivettiFaces()

load('olivettifacesoriginal.mat');

DATA = [];

for i = 1:400
    img = reshape(faces(:,i), [64,64]);
    
    
    DATA = cat(3, DATA, img);
end 

size(DATA)
assignin('base', 'DATA', DATA);
save('olivettifaces.mat', 'DATA');
clear;
end