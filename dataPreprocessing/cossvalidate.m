function cossvalidate
load fisheriris                              %# load iris dataset
groups = ismember(species,'setosa');         %# create a two-class problem

X = evalin('base','DATA')';
individuals = 15; 
picsPerInd = 11;
Y = zeros(1,individuals*picsPerInd)';
for i = 1:individuals
    Y((i-1)*picsPerInd + 1:i*picsPerInd) = i;
end

t = templateSVM('Standardize',1,'KernelFunction','gaussian');

Mdl = fitcecoc(X,Y,'Learners',t, 'Verbose',2);

Mdl.ClassNames
CodingMat = Mdl.CodingMatrix

CVMdl = crossval(Mdl);

oosLoss = kfoldLoss(CVMdl)


%# number of cross-validation folds:
%# If you have 50 samples, divide them into 10 groups of 5 samples each,
%# then train with 9 groups (45 samples) and test with 1 group (5 samples).
%# This is repeated ten times, with each group used exactly once as a test set.
%# Finally the 10 results from the folds are averaged to produce a single 
%# performance estimation.
k=10;

cvFolds = crossvalind('Kfold', groups, k);   %# get indices of 10-fold CV
cp = classperf(groups);                      %# init performance tracker

for i = 1:k                                  %# for each fold
    testIdx = (cvFolds == i);                %# get indices of test instances
    trainIdx = ~testIdx;                     %# get indices training instances

    %# train an SVM model over training instances
    svmModel = svmtrain(meas(trainIdx,:), groups(trainIdx), ...
                 'Autoscale',true, 'Showplot',false, 'Method','QP', ...
                 'BoxConstraint',2e-1, 'Kernel_Function','rbf', 'RBF_Sigma',1);

    %# test using test instances
    pred = svmclassify(svmModel, meas(testIdx,:), 'Showplot',false);

    %# evaluate and update performance object
    cp = classperf(cp, pred, testIdx);
end

%# get accuracy
cp.CorrectRate

%# get confusion matrix
%# columns:actual, rows:predicted, last-row: unclassified instances
cp.CountingMatrix

end