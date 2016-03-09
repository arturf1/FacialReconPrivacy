function faceRec()
 
individuals = 15; 
picsPerInd = 11;
X = evalin('base','X');

filterSize = []
RR = []

for j = 5:5:45
    correctRecog = 0;
    featEng(j)
    X = evalin('base','X');
    
    for randInd = 1:individuals
        for randPic = 1:picsPerInd

            LABELS = zeros(size(X,2),1);
            LABELS((randInd-1)*picsPerInd+1:(randInd-1)*picsPerInd+picsPerInd) = 1;

            TL = LABELS((randInd-1)*picsPerInd+randPic);
            T = X(:,(randInd-1)*picsPerInd+randPic);
            X_temp = X;
            X_temp(:,(randInd-1)*picsPerInd+randPic) = [];
            LABELS((randInd-1)*picsPerInd+randPic) = [];

            SVM = fitcsvm(X_temp',LABELS,'Standardize',true,'KernelFunction','RBF','KernelScale','auto');

            if predict(SVM, T') == TL
                correctRecog = correctRecog + 1;
            end
            strcat('Person  ', num2str(randInd), ' Pic: ', num2str(randPic))
        end
    end

    correctRecog/size(X,2)
    RR = [RR correctRecog/size(X,2)];
    s = ((100-j)*320/100 - floor(j*320/100)) * ((100-j)*243/100 - floor(j*243/100));
    filterSize = [filterSize s];
end

figure;
scatter(filterSize,RR)
filterSize
RR
xlabel('Number of Features');
ylabel('Recognition Rate');
title('Effect of Frequency Filtering on Recognition Rate');

end