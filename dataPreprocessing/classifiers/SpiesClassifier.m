function SpiesClassifier()

% YaleFace
individuals = 15; 
picsPerInd = 11;
load('SpiesFMyalefaces.mat')
X = SpiesFM;
correctRecog = 0;

for randInd = 1:individuals
    for randPic = 1:picsPerInd

        LABELS = zeros(size(X,2),1);
        LABELS((randInd-1)*picsPerInd+1:(randInd-1)*picsPerInd+picsPerInd) = 1;

        TL = LABELS((randInd-1)*picsPerInd+randPic);
        T = X(:,(randInd-1)*picsPerInd+randPic);
        X_temp = X;
        X_temp(:,(randInd-1)*picsPerInd+randPic) = [];

        dist = [];
        
        for i = 1:size(X_temp,2)
            dist = [dist norm(T - X_temp(:,i))];
        end
        
        [mini, argmini] = min(dist);
        
        if LABELS(argmini) == TL
            correctRecog = correctRecog + 1;
        end
        strcat('Person  ', num2str(randInd), ' Pic: ', num2str(randPic));
    end
end

correctRecog/size(X,2)

% OlivettiFace
individuals = 40; 
picsPerInd = 10;
load('SpiesFMolivettifaces.mat')
X = SpiesFM;
correctRecog = 0;

for randInd = 1:individuals
    for randPic = 1:picsPerInd

        LABELS = zeros(size(X,2),1);
        LABELS((randInd-1)*picsPerInd+1:(randInd-1)*picsPerInd+picsPerInd) = 1;

        TL = LABELS((randInd-1)*picsPerInd+randPic);
        T = X(:,(randInd-1)*picsPerInd+randPic);
        X_temp = X;
        X_temp(:,(randInd-1)*picsPerInd+randPic) = [];

        dist = [];
        
        for i = 1:size(X_temp,2)
            dist = [dist norm(T - X_temp(:,i))];
        end
        
        [mini, argmini] = min(dist);
        
        if LABELS(argmini) == TL
            correctRecog = correctRecog + 1;
        end
        strcat('Person  ', num2str(randInd), ' Pic: ', num2str(randPic));
    end
end

correctRecog/size(X,2)
end